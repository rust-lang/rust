//@ run-pass
use foo::bar;
mod foo {
    pub use crate::foo::zed::bar;
    pub mod zed {
        pub fn bar() { println!("foo"); }
    }
}

pub fn main() { bar(); }
