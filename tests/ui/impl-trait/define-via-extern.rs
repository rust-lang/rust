#![feature(type_alias_impl_trait)]

type Hi = impl Sized;

extern "C" {
    #[define_opaque(Hi)] fn foo();
    //~^ ERROR only functions, statics, and consts can define opaque types

    #[define_opaque(Hi)] static HI: Hi;
    //~^ ERROR only functions, statics, and consts can define opaque types
}

#[define_opaque(Hi)]
fn main() {
    let _: Hi = 0;
}
