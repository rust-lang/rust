// The canonical query `Projection(<get_rpit as FnOnce>::Output = Opaque)`
// is the *only* site that defines `Opaque` in MIR typeck.
//
// check-pass

#![feature(type_alias_impl_trait)]

type Opaque = impl Sized;

fn get_rpit() -> impl Sized {}

fn query(_: impl FnOnce() -> Opaque) {}

fn test(_: Opaque) {
    query(get_rpit);
}

fn main() {}
