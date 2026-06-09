// Regression test for <https://github.com/rust-lang/rust/issues/97432>.

#![no_std]

mod inner {
    //@ set trait_id = "$.index[?(@.name=='Trait')].id"
    pub trait Trait {}
}

//@ set export_id = "$.index[?(@.docs=='First re-export')].id"
//@ is "$.index[?(@.inner.use.name=='Trait')].inner.use.id" $trait_id
/// First re-export
pub use inner::Trait;
//@ set reexport_id = "$.index[?(@.docs=='Second re-export')].id"
//@ is "$.index[?(@.inner.use.name=='Reexport')].inner.use.id" $trait_id
/// Second re-export
pub use inner::Trait as Reexport;

//@ ismany "$.index[?(@.name=='same_type_reexported_more_than_once')].inner.module.items[*]" $reexport_id $export_id
