//@ run-pass
#![allow(unused_imports)]

use baz::zed;
use baz::zed::bar;

mod baz {
    pub mod zed {
        pub fn bar() { println!("bar2"); }
    }
}

pub fn main() { bar(); }
