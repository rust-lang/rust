#![feature(type_alias_impl_trait)]
// check-pass

#[derive(Copy, Clone)]
struct Foo((u32, u32));

fn main() {
    type T = impl Copy;
    let foo: T = Foo((1u32, 2u32));
    let x = move || {
        let Foo((a, b)) = foo;
    };
}
