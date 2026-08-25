// https://github.com/rust-lang/rust/issues/100252

mod bar {
    //@ set baz = "$.index[?(@.name == 'Baz')].id"
    pub struct Baz;
    //@ set impl = "$.index[?(@.docs == 'impl')].id"
    /// impl
    impl Baz {
        //@ set doit = "$.index[?(@.name == 'doit')].id"
        pub fn doit() {}
    }
}

//@ set import = "$.index[?(@.inner.use)].id"
pub use bar::Baz;

//@ is "$.index[*].inner.module.items[*]" $import
//@ is "$.index[*].inner.use.id" $baz
//@ has "$.index[?(@.name == 'Baz')].inner.struct.impls[*]" $impl
//@ is "$.index[?(@.docs=='impl')].inner.impl.items[*]" $doit
