// edition: 2021
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(async_fn_in_trait)]
#![allow(incomplete_features)]

trait MyTrait {
    async fn foo_recursive(&self, n: usize) -> i32;
}

impl MyTrait for i32 {
    async fn foo_recursive(&self, n: usize) -> i32 {
        //~^ ERROR recursion in an `async fn` requires boxing
        if n > 0 {
            self.foo_recursive(n - 1).await
        } else {
            *self
        }
    }
}

fn main() {}
