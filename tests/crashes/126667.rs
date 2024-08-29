//@ known-bug: rust-lang/rust#126667
#![warn(rust_2021_compatibility)]

trait Static<'a> {}

struct Foo((u32, u32));

fn main() {
    type T = impl Static;
    let foo: T = Foo((1u32, 2u32));
    let x = move || {
        let Foo((a, b)) = foo;
    };
}
