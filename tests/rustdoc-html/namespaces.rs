// issue #34843: rustdoc prioritises documenting reexports from the type namespace

mod inner {
    pub mod sync {
        pub struct SomeStruct;
    }

    pub fn sync() {}
}

//@ has namespaces/sync/index.html
//@ has namespaces/fn.sync.html
//@ has namespaces/index.html '//a/@href' 'sync/index.html'
//@ has - '//a/@href' 'fn.sync.html'
#[doc(inline)]
pub use inner::sync;
