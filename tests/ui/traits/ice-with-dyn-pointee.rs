// run-pass
#![feature(ptr_metadata)]
// Address issue #112737 -- ICE with dyn Pointee
extern crate core;
use core::ptr::Pointee;

fn raw_pointer_in(_: &dyn Pointee<Metadata = ()>) {}

fn main() {
    raw_pointer_in(&42)
}
