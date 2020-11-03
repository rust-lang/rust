// run-pass
// ignore-emscripten no subprocess support

#![feature(set_stdio)]

use std::fmt;
use std::fmt::{Display, Formatter};
use std::io::set_panic;
use std::sync::{Arc, Mutex};

pub struct A;

impl Display for A {
    fn fmt(&self, _f: &mut Formatter<'_>) -> fmt::Result {
        panic!();
    }
}

fn main() {
    set_panic(Some(Arc::new(Mutex::new(Vec::new()))));
    assert!(std::panic::catch_unwind(|| {
        eprintln!("{}", A);
    })
    .is_err());
}
