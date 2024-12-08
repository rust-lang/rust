//@ edition: 2021
//@ check-pass

use std::future::Future;
use std::marker::PhantomData;

trait Lockable<K, V> {
    #[allow(async_fn_in_trait)]
    async fn lock_all_entries(&self) -> impl Future<Output = Guard<'_>>;
}

struct Guard<'a>(PhantomData<&'a ()>);

fn main() {}
