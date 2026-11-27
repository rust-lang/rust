// Regression test for https://github.com/rust-lang/rust/issues/152936

use std::collections::hash_map::{HashMap, Keys};
use std::marker::PhantomData;

trait MapAssertion<'a, K, V, R> {
    fn key_set(&self) -> Subject<Keys<K, V>, (), R>;
}

struct Subject<'a, T, V, R>(PhantomData<(&'a T, V, R)>);

impl<'a, K, V, R> MapAssertion<'a, K, V, R> for Subject<'a, HashMap<K, V>, (), R> {
    fn key_set(&self) -> Subject<'static, Keys<K, V>, (), R> {
        //~^ ERROR cannot infer an appropriate lifetime for lifetime parameter '_ in generic type due to conflicting requirements
        //~| ERROR mismatched types
    }
}

fn main() {}
