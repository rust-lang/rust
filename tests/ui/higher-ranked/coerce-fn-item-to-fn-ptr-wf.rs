// Regression test from litemap: reifying a generic higher-ranked fn item to a
// fn pointer must not require the generic parameters to outlive the (fresh)
// region of the reified signature.

//@ check-pass

use std::iter::Map;
use std::slice::Iter;

type KvIter<'a, K, V> = Map<Iter<'a, (K, V)>, for<'b> fn(&'b (K, V)) -> (&'b K, &'b V)>;

fn map_f<K, V>(input: &(K, V)) -> (&K, &V) {
    (&input.0, &input.1)
}

fn get_iter<'a, K, V>(slice: &'a [(K, V)]) -> KvIter<'a, K, V> {
    slice.iter().map(map_f)
}

fn main() {}
