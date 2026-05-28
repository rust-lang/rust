//@ check-pass

#![feature(type_alias_impl_trait)]
type Opq = impl Sized;
#[define_opaque(Opq)]
fn test() -> impl Iterator<Item = Opq> {
    Box::new(0..) as Box<dyn Iterator<Item = _>>
}
fn main() {}
