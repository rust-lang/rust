//@ check-pass
//@ aux-build:implicit-pub.rs
//@ edition:2018

extern crate implicit_pub;

fn main() {
    implicit_pub::inner::real_pub!();
    implicit_pub::real_pub_reexport!();
}
