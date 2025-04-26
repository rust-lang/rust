#![crate_name = "foo"]

mod foo {
    pub use bar::*;
    pub mod bar {
        pub trait Foo {
            fn foo();
        }
    }
}

//@ count foo/index.html '//*[@class="trait"]' 1
pub use foo::bar::*;
pub use foo::*;
