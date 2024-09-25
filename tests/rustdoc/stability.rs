#![feature(staged_api)]

#![stable(feature = "rust1", since = "1.0.0")]

//@ has stability/index.html
//@ has - '//ul[@class="item-table"]/li[1]//a' AaStable
//@ has - '//ul[@class="item-table"]/li[2]//a' ZzStable
//@ has - '//ul[@class="item-table"]/li[3]//a' Unstable

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
    //@ !hasraw stability/unstable/struct.Foo.html '//span[@class="since"]'
    //@ has - '//div[@class="stab unstable"]' 'experimental'
    #[stable(feature = "rust1", since = "1.0.0")]
    pub struct Foo;
}

#[stable(feature = "rust2", since = "2.2.2")]
pub mod stable_later {
    //@ has stability/stable_later/struct.Bar.html '//span[@class="since"]' '2.2.2'
    #[stable(feature = "rust1", since = "1.0.0")]
    pub struct Bar;
}

#[stable(feature = "rust1", since = "1.0.0")]
pub mod stable_earlier {
    //@ has stability/stable_earlier/struct.Foo.html '//span[@class="since"]' '1.0.0'
    #[doc(inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use crate::unstable::Foo;

    //@ has stability/stable_earlier/struct.Bar.html '//span[@class="since"]' '1.0.0'
    #[doc(inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use crate::stable_later::Bar;
}
