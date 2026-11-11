//! Check that panics in `Display::fmt` during printing are properly handled.

//@ run-pass
//@ needs-unwind

#![feature(internal_output_capture)]

use std::fmt;
use std::fmt::{Display, Formatter};
use std::io::set_output_capture;
use std::sync::{Arc, Mutex};

pub struct A;

impl Display for A {
    fn fmt(&self, _f: &mut Formatter<'_>) -> fmt::Result {
        panic!();
    }
}

fn main() {
    set_output_capture(Some(Arc::new(Mutex::new(Vec::new()))));
    assert!(
        std::panic::catch_unwind(|| {
            eprintln!("{}", A);
        })
        .is_err()
    );
}
