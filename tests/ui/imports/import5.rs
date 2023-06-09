// run-pass
use foo::bar;
mod foo {
    pub use foo::zed::bar;
    pub mod zed {
        pub fn bar() { println!("foo"); }
    }
}

pub fn main() { bar(); }
