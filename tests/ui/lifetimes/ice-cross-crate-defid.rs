//@ aux-build:ice_trait.rs
//@ check-fail
extern crate ice_trait;
use ice_trait::ExternalTrait;
fn test<'a, T: ExternalTrait>(a: &'a mut T) -> impl std::future::Future<Output = ()> + 'static {
    a.build_request()
}
fn main() {}
