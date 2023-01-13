// ignore-tidy-linelength

// Regression test for <https://github.com/rust-lang/rust/issues/97432>.

#![feature(no_core)]
#![no_std]
#![no_core]

mod inner {
    // @set trait_id = "$.index[*][?(@.name=='Trait')].id"
    pub trait Trait {}
}

// @set export_id = "$.index[*][?(@.inner.name=='Trait')].id"
// @is "$.index[*][?(@.inner.name=='Trait')].inner.id" $trait_id
pub use inner::Trait;
// @set reexport_id = "$.index[*][?(@.inner.name=='Reexport')].id"
// @is "$.index[*][?(@.inner.name=='Reexport')].inner.id" $trait_id
pub use inner::Trait as Reexport;

// @ismany "$.index[*][?(@.name=='same_type_reexported_more_than_once')].inner.items[*]" $reexport_id $export_id
