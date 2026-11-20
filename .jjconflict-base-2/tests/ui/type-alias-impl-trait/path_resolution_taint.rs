//! This test used to ICE #131298

#![feature(type_alias_impl_trait)]

fn dyn_hoops<T>() -> *const dyn Iterator<Item = impl Captures> {
    //~^ ERROR: cannot find trait `Captures` in this scope
    loop {}
}

type Opaque = impl Sized;
#[define_opaque(Opaque)]
fn define() -> Opaque {
    let _: Opaque = dyn_hoops::<u8>();
}

fn main() {}
