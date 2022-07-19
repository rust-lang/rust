// Regression test for <https://github.com/rust-lang/rust/issues/97432>.

#![feature(no_core)]
#![no_std]
#![no_core]

// @has same_type_reexported_more_than_once.json
// @has - "$.index[*][?(@.name=='Trait')]"
pub use inner::Trait;
// @has - "$.index[*].inner[?(@.name=='Reexport')].id"
pub use inner::Trait as Reexport;

mod inner {
    pub trait Trait {}
}
