//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::Index;

struct HashMap<K, V>(PhantomData<(K, V)>);

impl<K, V> Index<&K> for HashMap<K, V>
where
    K: Hash,
    V: Copy,
{
    type Output = V;

    fn index(&self, k: &K) -> &V {
        todo!()
    }
}

fn index<'a, K, V>(map: &'a HashMap<K, V>, k: K) -> &'a V {
    map[k]
    //~^ ERROR the trait bound `K: Hash` is not satisfied
    //~| ERROR the trait bound `V: Copy` is not satisfied
    //~| ERROR mismatched types
    //[current]~| ERROR mismatched types
    //[next]~^^^^^ ERROR the trait bound `K: Hash` is not satisfied
}

fn main() {}
