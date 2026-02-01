//@ aux-build:implicit-pub.rs
//@ edition:2018

extern crate implicit_pub;

fn main() {
    implicit_pub::inner::fake_pub!(); //~ error: failed to resolve: could not find `fake_pub` in `inner` [E0433]
}
