//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(dyn_star)]
#![allow(incomplete_features)]

use std::fmt::Debug;

fn dyn_debug(_: (dyn* Debug + '_)) {

}

fn polymorphic<T: Debug + ?Sized>(t: &T) {
    dyn_debug(t);
    //~^ ERROR `&T` needs to have the same ABI as a pointer
}

fn main() {}
