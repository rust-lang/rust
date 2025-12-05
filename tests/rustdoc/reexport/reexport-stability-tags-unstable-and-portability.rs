#![crate_name = "foo"]
#![feature(doc_cfg)]
#![feature(staged_api)]
#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
pub mod tag {
    #[unstable(feature = "humans", issue = "none")]
    pub trait Unstable {}

    #[stable(feature = "rust1", since = "1.0.0")]
    #[doc(cfg(feature = "sync"))]
    pub trait Portability {}

    #[unstable(feature = "humans", issue = "none")]
    #[doc(cfg(feature = "sync"))]
    pub trait Both {}

    #[stable(feature = "rust1", since = "1.0.0")]
    pub trait None {}
}

//@ has foo/mod1/index.html
#[stable(feature = "rust1", since = "1.0.0")]
pub mod mod1 {
    //@ has - '//code' 'pub use tag::Unstable;'
    //@ has - '//span' 'Experimental'
    //@ !has - '//span' 'sync'
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use tag::Unstable;
}

//@ has foo/mod2/index.html
#[stable(feature = "rust1", since = "1.0.0")]
pub mod mod2 {
    //@ has - '//code' 'pub use tag::Portability;'
    //@ !has - '//span' 'Experimental'
    //@ !has - '//span' 'sync'
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use tag::Portability;
}

//@ has foo/mod3/index.html
#[stable(feature = "rust1", since = "1.0.0")]
pub mod mod3 {
    //@ has - '//code' 'pub use tag::Both;'
    //@ has - '//span' 'Experimental'
    //@ !has - '//span' 'sync'
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use tag::Both;
}

//@ has foo/mod4/index.html
#[stable(feature = "rust1", since = "1.0.0")]
pub mod mod4 {
    //@ has - '//code' 'pub use tag::None;'
    //@ !has - '//span' 'Experimental'
    //@ !has - '//span' 'sync'
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use tag::None;
}
