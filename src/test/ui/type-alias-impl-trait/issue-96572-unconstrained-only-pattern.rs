#![feature(type_alias_impl_trait)]
// check-pass

type T = impl Copy;

fn foo(foo: T) {
    let (mut x, mut y) = foo;
    x = 42;
    y = "foo";
}

fn main() {}
