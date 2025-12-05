//@ run-pass
#![allow(unused_imports)]

use foo::zed;
use bar::baz;

mod foo {
    pub mod zed {
        pub fn baz() { println!("baz"); }
    }
}
mod bar {
    pub use crate::foo::zed::baz;
}
pub fn main() { baz(); }
