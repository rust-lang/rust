#![feature(type_alias_impl_trait)]

pub trait Backend {}

impl Backend for () {}

pub struct Module<T>(T);

pub type BackendImpl = impl Backend;

//@ has return_impl_trait/fn.make_module.html
/// Documentation
pub fn make_module() -> Module<BackendImpl> {
    Module(())
}
