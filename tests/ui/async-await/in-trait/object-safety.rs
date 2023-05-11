// edition:2021
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(async_fn_in_trait)]

trait Foo {
    async fn foo(&self);
}

fn main() {
    let x: &dyn Foo = todo!();
    //~^ ERROR the trait `Foo` cannot be made into an object
}
