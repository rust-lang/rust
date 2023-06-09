// https://github.com/rust-lang/rust/issues/100252

#![feature(no_core)]
#![no_core]

mod bar {
    // @set baz = "$.index[*][?(@.inner.struct)].id"
    pub struct Baz;
    // @set impl = "$.index[*][?(@.inner.impl)].id"
    impl Baz {
        // @set doit = "$.index[*][?(@.inner.function)].id"
        pub fn doit() {}
    }
}

// @set import = "$.index[*][?(@.inner.import)].id"
pub use bar::Baz;

// @is "$.index[*].inner.module.items[*]" $import
// @is "$.index[*].inner.import.id" $baz
// @is "$.index[*].inner.struct.impls[*]" $impl
// @is "$.index[*].inner.impl.items[*]" $doit
