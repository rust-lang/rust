// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir

//! This test ensures that a mutable reference cannot be passed as a resume argument twice.

#![feature(generators, generator_trait)]

use std::marker::Unpin;
use std::ops::{
    Generator,
    GeneratorState::{self, *},
};
use std::pin::Pin;

fn main() {
    let mut thing = String::from("hello");

    let mut gen = |r| {
        if false {
            yield r;
        }
    };

    let mut gen = Pin::new(&mut gen);
    gen.as_mut().resume(&mut thing);
    gen.as_mut().resume(&mut thing);
    //~^ cannot borrow `thing` as mutable more than once at a time
}
