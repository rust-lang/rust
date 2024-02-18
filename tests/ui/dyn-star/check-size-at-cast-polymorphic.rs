//@ check-pass

#![feature(dyn_star)]
#![allow(incomplete_features)]

use std::fmt::Debug;

fn dyn_debug(_: (dyn* Debug + '_)) {

}

fn polymorphic<T: Debug>(t: &T) {
    dyn_debug(t);
}

fn main() {}
