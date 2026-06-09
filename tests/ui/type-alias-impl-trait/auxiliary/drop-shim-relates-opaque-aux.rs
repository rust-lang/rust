// crate foo

#![feature(type_alias_impl_trait)]

type Tait = impl Sized;
#[define_opaque(Tait)]
fn _constrain() -> Tait {}

struct WrapperWithDrop<T>(T);
impl<T> Drop for WrapperWithDrop<T> {
    fn drop(&mut self) {}
}

pub struct Foo(WrapperWithDrop<Tait>);

trait Id {
    type Id: ?Sized;
}
impl<T: ?Sized> Id for T {
    type Id = T;
}
pub struct Bar(WrapperWithDrop<<Tait as Id>::Id>);
