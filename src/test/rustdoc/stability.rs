#![feature(staged_api)]

#![unstable(feature = "test", issue = "none")]

pub struct Unstable {
    // @has stability/struct.Unstable.html \
    //      '//div[@class="item-info"]//div[@class="stab unstable"]' \
    //      'This is a nightly-only experimental API'
    // @count stability/struct.Unstable.html '//span[@class="stab unstable"]' 0
    pub foo: u32,
    pub bar: u32,
}

#[unstable(feature = "test", issue = "1")]
#[unstable(feature = "test2", issue = "2")]
pub trait UnstableTrait {
    // @has stability/trait.UnstableTrait.html \
    //      '//div[@class="item-info"]//div[@class="stab unstable"]' \
    //      'This is a nightly-only experimental API. (test, test2)'
}
