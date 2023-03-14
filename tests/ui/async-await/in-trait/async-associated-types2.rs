// check-pass
// edition: 2021
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(async_fn_in_trait)]
#![feature(type_alias_impl_trait)]
#![allow(incomplete_features)]

use std::future::Future;

trait MyTrait {
    type Fut<'a>: Future<Output = i32>
    where
        Self: 'a;

    fn foo<'a>(&'a self) -> Self::Fut<'a>;
}

impl MyTrait for i32 {
    type Fut<'a> = impl Future<Output = i32> + 'a
    where
        Self: 'a;

    fn foo<'a>(&'a self) -> Self::Fut<'a> {
        async {
            *self
        }
    }
}

fn main() {}
