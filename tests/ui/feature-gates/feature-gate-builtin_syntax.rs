//@ edition: 2021..
#![feature(forced_keywords)]

struct Foo {
    v: u8,
    w: u8,
}
fn main() {
    k#offset_of(Foo, v); //~ ERROR `builtin #` syntax is unstable
}
