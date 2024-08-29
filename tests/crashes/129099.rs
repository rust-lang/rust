//@ known-bug: rust-lang/rust#129099

#![feature(type_alias_impl_trait)]

fn dyn_hoops<T: Sized>() -> dyn for<'a> Iterator<Item = impl Captures<'a>> {
    loop {}
}

pub fn main() {
    type Opaque = impl Sized;
    fn define() -> Opaque {
        let x: Opaque = dyn_hoops::<()>(0);
        x
    }
}
