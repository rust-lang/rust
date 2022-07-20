// compile-flags: --edition=2021
#![feature(type_alias_impl_trait)]

fn main() {
    type T = impl Copy; //~ ERROR unconstrained opaque type
    let foo: T = (1u32, 2u32);
    let (a, b): (u32, u32) = foo;
}
