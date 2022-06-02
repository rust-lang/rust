// Regression test for https://github.com/rust-lang/rust/issues/97432.

#![feature(no_core)]
#![no_std]
#![no_core]

// @has same_type_reexported_more_than_once.json
// @set trait_id = - "$.index[*][?(@.name=='Trait')].id"
// @has - "$.index[*][?(@.name=='same_type_reexported_more_than_once')].inner.items[*]" $trait_id
pub use inner::Trait;
// @set reexport_id = - "$.index[*][?(@.name=='Reexport')].id"
// @has - "$.index[*][?(@.name=='same_type_reexported_more_than_once')].inner.items[*]" $reexport_id
pub use inner::Trait as Reexport;

mod inner {
    pub trait Trait {}
}
