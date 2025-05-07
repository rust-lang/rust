#![feature(staged_api)]
#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]

#![stable(feature = "core", since = "1.6.0")]

//@ has stability/index.html
//@ has - '//dl[@class="item-table"]/dt[1]//a' AaStable
//@ has - '//dl[@class="item-table"]/dt[2]//a' ZzStable
//@ has - '//dl[@class="item-table"]/dt[3]//a' Unstable

#[stable(feature = "rust2", since = "2.2.2")]
pub struct AaStable;

#[unstable(feature = "test", issue = "none")]
pub struct Unstable {
    //@ has stability/struct.Unstable.html \
    //      '//span[@class="item-info"]//div[@class="stab unstable"]' \
    //      'This is a nightly-only experimental API'
    //@ count stability/struct.Unstable.html '//span[@class="stab unstable"]' 0
    pub foo: u32,
    pub bar: u32,
}

#[stable(feature = "rust2", since = "2.2.2")]
pub struct ZzStable;

#[unstable(feature = "unstable", issue = "none")]
pub mod unstable {
    //@ !hasraw stability/unstable/struct.StableInUnstable.html \
    //      '//div[@class="main-heading"]//span[@class="since"]'
    //@ has - '//div[@class="stab unstable"]' 'experimental'
    #[stable(feature = "rust1", since = "1.0.0")]
    pub struct StableInUnstable;

    #[stable(feature = "rust1", since = "1.0.0")]
    pub mod stable_in_unstable {
        //@ !hasraw stability/unstable/stable_in_unstable/struct.Inner.html \
        //      '//div[@class="main-heading"]//span[@class="since"]'
        //@ has - '//div[@class="stab unstable"]' 'experimental'
        #[stable(feature = "rust1", since = "1.0.0")]
        pub struct Inner;
    }

    //@ has stability/struct.AaStable.html \
    //      '//*[@id="method.foo"]//span[@class="since"]' '2.2.2'
    impl super::AaStable {
        #[stable(feature = "rust2", since = "2.2.2")]
        pub fn foo() {}
    }

    //@ has stability/unstable/struct.StableInUnstable.html \
    //      '//*[@id="method.foo"]//span[@class="since"]' '1.0.0'
    impl StableInUnstable {
        #[stable(feature = "rust1", since = "1.0.0")]
        pub fn foo() {}
    }
}

#[unstable(feature = "unstable", issue = "none")]
#[doc(hidden)]
pub mod unstable_stripped {
    //@ has stability/struct.AaStable.html \
    //      '//*[@id="method.foo"]//span[@class="since"]' '2.2.2'
    impl super::AaStable {
        #[stable(feature = "rust2", since = "2.2.2")]
        pub fn foo() {}
    }
}

#[stable(feature = "rust2", since = "2.2.2")]
pub mod stable_later {
    //@ has stability/stable_later/struct.StableInLater.html \
    //      '//div[@class="main-heading"]//span[@class="since"]' '2.2.2'
    #[stable(feature = "rust1", since = "1.0.0")]
    pub struct StableInLater;

    #[stable(feature = "rust1", since = "1.0.0")]
    pub mod stable_in_later {
        //@ has stability/stable_later/stable_in_later/struct.Inner.html \
        //      '//div[@class="main-heading"]//span[@class="since"]' '2.2.2'
        #[stable(feature = "rust1", since = "1.0.0")]
        pub struct Inner;
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_allowed_through_unstable_modules = "use stable path instead"]
pub mod stable_earlier1 {
    //@ has stability/stable_earlier1/struct.StableInUnstable.html \
    //      '//div[@class="main-heading"]//span[@class="since"]' '1.0.0'
    //@ has - '//*[@id="method.foo"]//span[@class="since"]' '1.0.0'
    #[doc(inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use crate::unstable::StableInUnstable;

    //@ has stability/stable_earlier1/stable_in_unstable/struct.Inner.html \
    //      '//div[@class="main-heading"]//span[@class="since"]' '1.0.0'
    #[doc(inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use crate::unstable::stable_in_unstable;

    //@ has stability/stable_earlier1/struct.StableInLater.html \
    //      '//div[@class="main-heading"]//span[@class="since"]' '1.0.0'
    #[doc(inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use crate::stable_later::StableInLater;

    //@ has stability/stable_earlier1/stable_in_later/struct.Inner.html \
    //      '//div[@class="main-heading"]//span[@class="since"]' '1.0.0'
    #[doc(inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use crate::stable_later::stable_in_later;
}

/// These will inherit the crate stability.
#[stable(feature = "rust1", since = "1.0.0")]
pub mod stable_earlier2 {
    //@ has stability/stable_earlier2/struct.StableInUnstable.html \
    //      '//div[@class="main-heading"]//span[@class="since"]' '1.6.0'
    //@ has - '//*[@id="method.foo"]//span[@class="since"]' '1.0.0'
    #[doc(inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use crate::unstable::StableInUnstable;

    //@ has stability/stable_earlier2/stable_in_unstable/struct.Inner.html \
    //      '//div[@class="main-heading"]//span[@class="since"]' '1.6.0'
    #[doc(inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use crate::unstable::stable_in_unstable;

    //@ has stability/stable_earlier2/struct.StableInLater.html \
    //      '//div[@class="main-heading"]//span[@class="since"]' '1.6.0'
    #[doc(inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use crate::stable_later::StableInLater;

    //@ has stability/stable_earlier2/stable_in_later/struct.Inner.html \
    //      '//div[@class="main-heading"]//span[@class="since"]' '1.6.0'
    #[doc(inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use crate::stable_later::stable_in_later;
}

//@ !hasraw stability/trait.UnstableTraitWithStableMethod.html \
//      '//div[@class="main-heading"]//span[@class="since"]'
//@ has - '//*[@id="tymethod.foo"]//span[@class="since"]' '1.0.0'
//@ has - '//*[@id="method.bar"]//span[@class="since"]' '1.0.0'
#[unstable(feature = "unstable", issue = "none")]
pub trait UnstableTraitWithStableMethod {
    #[stable(feature = "rust1", since = "1.0.0")]
    fn foo();
    #[stable(feature = "rust1", since = "1.0.0")]
    fn bar() {}
}

//@ has stability/primitive.i32.html \
//      '//div[@class="main-heading"]//span[@class="since"]' '1.0.0'
#[rustc_doc_primitive = "i32"]
//
/// `i32` is always stable in 1.0, even if you look at it from core.
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_i32 {}

//@ has stability/keyword.if.html \
//      '//div[@class="main-heading"]//span[@class="since"]' '1.0.0'
#[doc(keyword = "if")]
//
/// We currently don't document stability for keywords, but let's test it anyway.
#[stable(feature = "rust1", since = "1.0.0")]
mod if_keyword {}
