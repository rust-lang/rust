// build-pass

use std::fmt;

pub struct Wrapper(fn(val: &()));

impl fmt::Debug for Wrapper {
    fn fmt<'a>(&'a self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Wrapper").field(&self.0 as &fn(&'a ())).finish()
    }
}

fn useful(_: &()) {
}

fn main() {
    println!("{:?}", Wrapper(useful));
}
