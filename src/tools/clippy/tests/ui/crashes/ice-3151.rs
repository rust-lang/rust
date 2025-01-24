// Test for https://github.com/rust-lang/rust-clippy/issues/3151

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
