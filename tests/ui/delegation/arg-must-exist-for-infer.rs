#![feature(fn_delegation)]

struct S;

trait Trait<const Self: usize = 2> {
    //~^ ERROR expected identifier, found keyword `Self`
    fn foo();
}

impl S {
    reuse Trait::<_>::foo;
    //~^ ERROR type annotations needed
}

fn main() {}
