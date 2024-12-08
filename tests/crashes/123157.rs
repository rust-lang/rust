//@ known-bug: #123157
//@ edition:2021
#![feature(type_alias_impl_trait)]

#[derive(Copy, Clone)]
struct Foo((u32, u32));

fn main() {
    type T = impl Copy;
    let foo: T = Foo((1u32, 2u32));
    let x = move || {
        let x = move || {
        let Foo((a, b)) = foo;
    };
    };
}
