//@ edition: 2024
//@ aux-build: captured-args.rs

// Tests that we properly record cross-crate alias liveness.

extern crate captured_args;

use captured_args::Captures;

fn late() {
    let s = String::new();
    let val = captured_args::late(&s);
    drop(s);
    //~^ ERROR cannot move out of `s` because it is borrowed
    drop(val);
}

fn early() {
    let s = String::new();
    let val = captured_args::early(&s);
    drop(s);
    //~^ ERROR cannot move out of `s` because it is borrowed
    drop(val);
}

fn type_param() {
    let s = String::new();
    let val = captured_args::type_param(&s);
    drop(s);
    //~^ ERROR cannot move out of `s` because it is borrowed
    drop(val);
}

fn rpitit_late<T: Captures>() {
    let s = String::new();
    let val = T::late(&s);
    drop(s);
    //~^ ERROR cannot move out of `s` because it is borrowed
    drop(val);
}

fn rpitit_early<T: Captures>() {
    let s = String::new();
    let val = T::early(&s);
    drop(s);
    //~^ ERROR cannot move out of `s` because it is borrowed
    drop(val);
}

// An uncaptured argument must not keep either kind of alias live.
fn uncaptured<T: Captures>() {
    let mut s = String::new();
    let opaque = captured_args::uncaptured(&mut s);
    let rpitit = T::uncaptured(&mut s);
    drop(s);
    drop((opaque, rpitit));
}

fn main() {}
