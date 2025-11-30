#![feature(ptr_metadata)]
// Address issue #112737 -- ICE with dyn Pointee
extern crate core;
use core::ptr::Pointee;

fn unknown_sized_object_ptr_in(_: &(impl Pointee<Metadata = ()> + ?Sized)) {}

fn raw_pointer_in(x: &dyn Pointee<Metadata = ()>) {
    //~^ ERROR the trait `Pointee` is not dyn compatible
    unknown_sized_object_ptr_in(x)
    //~^ ERROR the trait `Pointee` is not dyn compatible
}

fn main() {
    raw_pointer_in(&42)
    //~^ ERROR the trait `Pointee` is not dyn compatible
}
