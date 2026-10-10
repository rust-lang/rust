// The canonical query `Projection(<get_rpit as FnOnce>::Output = Opaque)`
// is the *only* site that defines `Opaque` in MIR typeck.
//
//@ check-pass

#![feature(type_alias_impl_trait)]

pub type Opaque = impl Sized;

pub fn get_rpit() -> impl Sized {}

#[define_opaque(Opaque)]
fn test() {
    query(get_rpit);
}

fn query(_: impl FnOnce() -> Opaque) {}

fn main() {}
