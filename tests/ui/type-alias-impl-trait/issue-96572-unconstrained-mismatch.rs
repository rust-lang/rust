#![feature(type_alias_impl_trait)]

fn main() {
    type T = impl Copy;
    let foo: T = Some((1u32, 2u32));
    match foo {
        None => (),
        Some((a, b, c)) => (), //~ ERROR mismatched types
    }
}
