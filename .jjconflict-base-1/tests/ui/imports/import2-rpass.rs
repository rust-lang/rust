//@ run-pass

use zed::bar;

mod zed {
    pub fn bar() { println!("bar"); }
}

pub fn main() { bar(); }
