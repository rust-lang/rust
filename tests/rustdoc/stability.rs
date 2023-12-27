#![feature(staged_api)]

#![unstable(feature = "test", issue = "none")]

// @has stability/index.html
// @has - '//ul[@class="item-table"]/li[1]//a' AaStable
// @has - '//ul[@class="item-table"]/li[2]//a' ZzStable
// @has - '//ul[@class="item-table"]/li[3]//a' Unstable

#[stable(feature = "rust2", since = "2.2.2")]
pub struct AaStable;

pub struct Unstable {
    // @has stability/struct.Unstable.html \
    //      '//span[@class="item-info"]//div[@class="stab unstable"]' \
    //      'This is a nightly-only experimental API'
    // @count stability/struct.Unstable.html '//span[@class="stab unstable"]' 0
    pub foo: u32,
    pub bar: u32,
}

#[stable(feature = "rust2", since = "2.2.2")]
pub struct ZzStable;
