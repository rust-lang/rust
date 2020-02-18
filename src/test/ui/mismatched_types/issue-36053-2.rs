// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl
// Regression test for #36053. ICE was caused due to obligations
// being added to a special, dedicated fulfillment cx during
// a probe.

use std::iter::once;
fn main() {
    once::<&str>("str").fuse().filter(|a: &str| true).count();
    //~^ ERROR no method named `count`
    //~| ERROR type mismatch in closure arguments
}
