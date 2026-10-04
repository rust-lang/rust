#![crate_name = "foo"]
#![feature(doc_cfg)]

pub mod tag {
    #[deprecated(since = "0.1.8", note = "Use bar() instead")]
    pub trait Deprecated {}

    #[doc(cfg(feature = "sync"))]
    pub trait Portability {}

    #[deprecated(since = "0.1.8", note = "Use bar() instead")]
    #[doc(cfg(feature = "sync"))]
    pub trait Both {}

    pub trait None {}
}

//@ has foo/mod1/index.html
pub mod mod1 {
    //@ has - '//code' 'pub use tag::Deprecated;'
    //@ has - '//span' 'Deprecated'
    //@ !has - '//span' 'sync'
    pub use tag::Deprecated;
}

//@ has foo/mod2/index.html
pub mod mod2 {
    //@ has - '//code' 'pub use tag::Portability;'
    //@ !has - '//span' 'Deprecated'
    //@ !has - '//span' 'sync'
    pub use tag::Portability;
}

//@ has foo/mod3/index.html
pub mod mod3 {
    //@ has - '//code' 'pub use tag::Both;'
    //@ has - '//span' 'Deprecated'
    //@ !has - '//span' 'sync'
    pub use tag::Both;
}

//@ has foo/mod4/index.html
pub mod mod4 {
    //@ has - '//code' 'pub use tag::None;'
    //@ !has - '//span' 'Deprecated'
    //@ !has - '//span' 'sync'
    pub use tag::None;
}
