#[derive(Clone)]
pub struct HashMap<V, S> {
    hash_builder: S,
    table: RawTable<V>,
}

#[derive(Clone)]
pub struct RawTable<V> {
    size: usize,
    val: V,
}

fn main() {}
