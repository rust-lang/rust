//! Test suggetions to borrow generic arguments instead of moving when all predicates hold after
//! substituting in the reference type.
//@ run-rustfix
//@ aux-build:suggest-borrow-for-generic-arg-aux.rs

#![allow(unused_mut)]
extern crate suggest_borrow_for_generic_arg_aux as aux;

pub fn main() {
    let bar = aux::Bar;
    aux::foo(bar); //~ HELP borrow the value
    let _baa = bar; //~ ERROR use of moved value
    let mut bar = aux::Bar;
    aux::qux(bar); //~ HELP borrow the value
    let _baa = bar; //~ ERROR use of moved value
    let bar = aux::Bar;
    aux::bat(bar); //~ HELP borrow the value
    let _baa = bar; //~ ERROR use of moved value
    let mut bar = aux::Bar;
    aux::baz(bar); //~ HELP borrow the value
    let _baa = bar; //~ ERROR use of moved value
}
