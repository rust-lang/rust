//@ known-bug: #131535
#![feature(non_lifetime_binders)]
trait v0<> {}
fn kind  :(v0<'_, > impl for<v4> v0<'_, v2 = impl v0<v4> + '_>) {}
