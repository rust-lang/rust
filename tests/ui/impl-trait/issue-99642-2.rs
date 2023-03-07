// check-pass

#![feature(type_alias_impl_trait)]
type Opq = impl Sized;
fn test() -> impl Iterator<Item = Opq> {
    Box::new(0..) as Box<dyn Iterator<Item = _>>
}
fn main(){}
