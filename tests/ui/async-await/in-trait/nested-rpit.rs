// edition: 2021
// [current] known-bug: #105197
// [current] failure-status:101
// [current] dont-check-compiler-stderr
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// [next] check-pass
// revisions: current next

#![feature(async_fn_in_trait)]
#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

use std::future::Future;
use std::marker::PhantomData;

trait Lockable<K, V> {
    async fn lock_all_entries(&self) -> impl Future<Output = Guard<'_>>;
}

struct Guard<'a>(PhantomData<&'a ()>);

fn main() {}
