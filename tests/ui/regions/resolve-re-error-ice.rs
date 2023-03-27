// check-pass

// Allow this for now, can remove this UI test when this becomes a hard error.
#![allow(implied_bounds_entailment)]

use std::collections::hash_map::{Keys, HashMap};
use std::marker::PhantomData;

trait MapAssertion<'a, K, V, R> {
    fn key_set(&self) -> Subject<Keys<K, V>, (), R>;
}

struct Subject<'a, T, V, R>(PhantomData<(&'a T, V, R)>);

impl<'a, K, V, R> MapAssertion<'a, K, V, R> for Subject<'a, HashMap<K, V>, (), R>
{
    fn key_set(&self) -> Subject<'a, Keys<K, V>, (), R> {
        todo!()
    }
}

fn main() {}
