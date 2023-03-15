// Regression test for <https://github.com/rust-lang/rust/issues/108570>.
// If a `#[doc(hidden)]` item is re-exported or if a private item is re-exported
// with a `#[doc(hidden)]` re-export, it shouldn't complain.

// check-pass

#![crate_type = "lib"]

mod priv_mod {
    pub struct Foo;
    #[doc(hidden)]
    pub struct Bar;
}

#[doc(hidden)]
pub use priv_mod::Foo;
pub use priv_mod::Bar;
