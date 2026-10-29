use std::collections::hash_map::{Keys, HashMap};
use std::marker::PhantomData;

trait MapAssertion<'a, K, V, R> {
    fn key_set(&self) -> Subject<Keys<K, V>, (), R>;
}

struct Subject<'a, T, V, R>(PhantomData<(&'a T, V, R)>);

impl<'a, K, V, R> MapAssertion<'a, K, V, R> for Subject<'a, HashMap<K, V>, (), R>
{
    fn key_set(&self) -> Subject<'a, Keys<K, V>, (), R> {
        //~^ ERROR cannot infer an appropriate lifetime for lifetime parameter '_ in generic type due to conflicting requirements
        todo!()
    }
}

fn main() {}
