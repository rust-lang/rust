// aux-build:submodule-outer.rs
// edition:2018
#![deny(broken_intra_doc_links)]

extern crate bar as bar_;

// from https://github.com/rust-lang/rust/issues/60883
pub mod bar {
    pub use ::bar_::Bar;
}

// NOTE: we re-exported both `Foo` and `Bar` here,
// NOTE: so they are inlined and therefore we link to the current module.
// @has 'submodule_outer/trait.Foo.html' '//a[@href="../submodule_outer/bar/trait.Bar.html"]' 'Bar'
// @has 'submodule_outer/trait.Foo.html' '//a[@href="../submodule_outer/trait.Baz.html"]' 'Baz'
pub use ::bar_::{Foo, Baz};
