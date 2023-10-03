// revisions: current next
//[next] compile-flags: -Ztrait-solver=next
// edition:2021

#![feature(impl_trait_in_assoc_type)]

use std::future::Future;

trait Foo {
    type T;
    type Fut2: Future<Output = Self::T>; // ICE got triggered with traits other than Future here
    type Fut: Future<Output = Self::Fut2>;
    fn get_fut(&self) -> Self::Fut;
}

struct Implementor;

impl Foo for Implementor {
    type T = u64;
    type Fut2 = impl Future<Output = u64>;
    type Fut = impl Future<Output = Self::Fut2>;

    fn get_fut(&self) -> Self::Fut {
        //[current]~^ ERROR `{integer}` is not a future
        async move {
            //[next]~^ ERROR mismatched types
            42
            // 42 does not impl Future and rustc does actually point out the error,
            // but rustc used to panic.
            // Putting a valid Future here always worked fine.
        }
    }
}

fn main() {}
