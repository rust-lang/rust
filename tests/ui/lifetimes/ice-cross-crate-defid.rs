//@ aux-build:ice_trait.rs
extern crate ice_trait;
use ice_trait::ExternalTrait;
// Forcing a lifetime error here, The compiler will try to explain 
// This should forces it to look at the ExternalTrait definition in the other crate.
fn test<'a, T: ExternalTrait>(a: &'a mut T) -> impl std::future::Future<Output = ()> + 'static {
    a.build_request() 
}

fn main() {}
