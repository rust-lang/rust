//@ known-bug: rust-lang/rust#125843
#![feature(non_lifetime_binders)]
trait v0<> {}
fn kind  :(v3main impl for<v4> v0<'_, v2 = impl v0<v4> + '_>) {}
