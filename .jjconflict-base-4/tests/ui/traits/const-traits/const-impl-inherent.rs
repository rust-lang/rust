#![feature(const_trait_impl)]

//@ check-pass

struct Foo;

const impl Foo {
    fn bar() {}
}

const _: () = Foo::bar();

fn main() {
    Foo::bar();
}
