#![feature(return_position_impl_trait_in_trait)]

struct Wrapper<G: Send>(G);

trait Foo {
    fn bar() -> Wrapper<impl Sized>;
    //~^ ERROR `impl Sized` cannot be sent between threads safely
}

fn main() {}
