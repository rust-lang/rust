#![feature(const_fn)]

fn main() {}

union Foo {
    u: u32,
    i: i32,
}

const unsafe fn foo(u: u32) -> i32 {
    Foo { u }.i //~ ERROR unions in const fn are unstable
}
