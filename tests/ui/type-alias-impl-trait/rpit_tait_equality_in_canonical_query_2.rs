// The canonical query `Projection(<get_rpit as FnOnce>::Output = Opaque)`
// is the *only* site that defines `Opaque` in MIR typeck.
//
//@ check-pass

#![feature(type_alias_impl_trait)]

mod helper {
    pub type Opaque = impl Sized;

    pub fn get_rpit() -> impl Sized {}

    fn test(_: Opaque) {
        super::query(get_rpit);
    }
}
use helper::*;

fn query(_: impl FnOnce() -> Opaque) {}

fn main() {}
