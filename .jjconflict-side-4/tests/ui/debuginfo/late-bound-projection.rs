//@ build-pass
//@ compile-flags: -Cdebuginfo=2 --crate-type=rlib
// Fixes issue #94998

pub trait Trait {}

pub fn run(_: &dyn FnOnce(&()) -> Box<dyn Trait + '_>) {}
