//@ edition:2021
// https://github.com/rust-lang/rust/issues/117547

trait T {}

trait MyTrait {
    async fn foo() -> &'static impl T;
    //~^ ERROR the associated type `impl T` may not live long enough
}

fn main() {}
