//@ known-bug: #119716
#![feature(non_lifetime_binders)]
trait v0<v1> {}
fn kind  :(v3main impl for<v4> v0<'_, v2 = impl v0<v4> + '_>) {}
