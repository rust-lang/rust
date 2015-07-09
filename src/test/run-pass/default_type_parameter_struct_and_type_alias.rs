use std::marker::PhantomData;

trait TypeEq<A> {}
impl<A> TypeEq<A> for A {}

struct DeterministicHasher;
struct RandomHasher;


struct MyHashMap<K, V, H=DeterministicHasher> {
    data: PhantomData<(K, V, H)>
}

impl<K, V, H> MyHashMap<K, V, H> {
    fn new() -> MyHashMap<K, V, H> {
        MyHashMap { data: PhantomData }
    }
}

mod mystd {
    use super::{MyHashMap, RandomHasher};
    pub type HashMap<K, V, H=RandomHasher> = MyHashMap<K, V, H>;
}

fn try_me<H>(hash_map: mystd::HashMap<i32, i32, H>) {}

fn main() {
    let hash_map = mystd::HashMap::new();
    try_me(hash_map);
}
