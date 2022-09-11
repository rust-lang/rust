// https://github.com/rust-lang/rust/issues/100252

#![feature(no_core)]
#![no_core]

mod bar {
    // @set baz = "$.index[*][?(@.kind=='struct')].id"
    pub struct Baz;
    // @set impl = "$.index[*][?(@.kind=='impl')].id"
    impl Baz {
        // @set doit = "$.index[*][?(@.kind=='method')].id"
        pub fn doit() {}
    }
}

// @set import = "$.index[*][?(@.kind=='import')].id"
pub use bar::Baz;

// @is "$.index[*][?(@.kind=='module')].inner.items[*]" $import
// @is "$.index[*][?(@.kind=='import')].inner.id" $baz
// @is "$.index[*][?(@.kind=='struct')].inner.impls[*]" $impl
// @is "$.index[*][?(@.kind=='impl')].inner.items[*]" $doit
