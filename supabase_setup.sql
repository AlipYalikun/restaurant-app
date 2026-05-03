-- Run this in your Supabase SQL editor BEFORE running ingest_embeddings.py
-- It sets up the table and the similarity search function your RAG pipeline will call.

-- Step 2: Create the menu items table
create table if not exists menu_items (
  id              text primary key,        -- e.g. "dolan_appetizers_lentil_soup"
  restaurant      text not null,
  address         text,
  cuisine         text,
  category        text,
  name            text not null,
  price           numeric(8, 2),
  spicy           boolean default false,
  dietary_tags    text[]  default '{}',    -- e.g. ["vegetarian", "vegan"]
  variants        jsonb   default '[]',    -- e.g. [{"option": "With Beef", "price": 19.95}]
  notes           text,
  embedding_text  text,                    -- the natural language sentence that was embedded
  embedding       vector(768),            -- the actual vector from OpenAI
  created_at      timestamptz default now(),
  updated_at      timestamptz default now()
);

-- Step 3: Create an index so similarity search is fast
-- ivfflat is the right index type for pgvector at small-to-medium scale
create index if not exists menu_items_embedding_idx
  on menu_items
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 10);   -- use 10 lists for ~66 items (rule of thumb: sqrt(row_count))

-- Step 4: Create the similarity search function your RAG pipeline calls
-- This is what FastAPI calls when a user asks for a recommendation.
create or replace function match_menu_items(
  query_embedding  vector(768),   -- the embedded user query
  match_count      int     default 5,
  max_price        numeric default null,
  filter_spicy     boolean default null,
  filter_dietary   text[]  default null
)
returns table (
  id             text,
  name           text,
  category       text,
  price          numeric,
  spicy          boolean,
  dietary_tags   text[],
  variants       jsonb,
  notes          text,
  embedding_text text,
  similarity     float
)
language sql stable
as $$
  select
    m.id,
    m.name,
    m.category,
    m.price,
    m.spicy,
    m.dietary_tags,
    m.variants,
    m.notes,
    m.embedding_text,
    1 - (m.embedding <=> query_embedding) as similarity
  from menu_items m
  where
    -- optional price filter
    (max_price is null or m.price <= max_price or m.price is null)
    -- optional spicy filter (null = include both)
    and (filter_spicy is null or m.spicy = filter_spicy)
    -- optional dietary filter (null = include everything)
    and (filter_dietary is null or m.dietary_tags @> filter_dietary)
  order by m.embedding <=> query_embedding   -- cosine distance, ascending
  limit match_count;
$$;
