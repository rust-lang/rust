//@ check-pass

#![warn(clippy::repeat_once)]
#![allow(clippy::let_unit_value)]

trait Repeat {
    fn repeat(&self) {}
}

impl Repeat for usize {
    fn repeat(&self) {}
}

fn main() {
    let _ = 42.repeat();
}
