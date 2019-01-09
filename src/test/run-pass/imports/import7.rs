// run-pass
#![allow(unused_imports)]

use foo::zed;
use bar::baz;

mod foo {
    pub mod zed {
        pub fn baz() { println!("baz"); }
    }
}
mod bar {
    pub use foo::zed::baz;
    pub mod foo {
        pub mod zed {}
    }
}
pub fn main() { baz(); }
