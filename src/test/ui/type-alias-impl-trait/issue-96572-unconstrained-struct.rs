#![feature(type_alias_impl_trait)]
// check-pass

#[derive(Copy, Clone)]
struct Foo((u32, u32));

fn main() {
    type U = impl Copy;
    let foo: U = Foo((1u32, 2u32));
    let Foo((a, b)) = foo;
}
