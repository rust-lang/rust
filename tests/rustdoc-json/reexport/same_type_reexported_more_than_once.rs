// ignore-tidy-linelength

// Regression test for <https://github.com/rust-lang/rust/issues/97432>.

#![feature(no_core)]
#![no_std]
#![no_core]

mod inner {
    // @set trait_id = "$.index[*][?(@.name=='Trait')].id"
    pub trait Trait {}
}

// @set export_id = "$.index[*][?(@.docs=='First re-export')].id"
// @is "$.index[*].inner[?(@.import.name=='Trait')].import.id" $trait_id
/// First re-export
pub use inner::Trait;
// @set reexport_id = "$.index[*][?(@.docs=='Second re-export')].id"
// @is "$.index[*].inner[?(@.import.name=='Reexport')].import.id" $trait_id
/// Second re-export
pub use inner::Trait as Reexport;

// @ismany "$.index[*][?(@.name=='same_type_reexported_more_than_once')].inner.module.items[*]" $reexport_id $export_id
