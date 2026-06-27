// TODO: What to do with this test?

pub trait Indexable {
    type Idx;
}
impl Indexable for u8 {
    type Idx = u8;
}
impl Indexable for u16 {
    type Idx = u16;
}

pub trait Indexer<T: Indexable>: std::ops::Index<T::Idx, Output = T> {}

trait StoreIndex: Indexer<u8> + Indexer<u16> {}

fn foo(st: &impl StoreIndex) -> &dyn StoreIndex {
    //~^ ERROR conflicting associated type bindings for `Output`
    st as &dyn StoreIndex
    //~^ ERROR conflicting associated type bindings for `Output`
}

fn main() {}
