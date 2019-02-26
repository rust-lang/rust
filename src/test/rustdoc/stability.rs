#![feature(staged_api)]

#![unstable(feature = "test", issue = "0")]

pub struct Unstable {
    // @has stability/struct.Unstable.html \
    //      '//div[@class="stability"]//div[@class="stab unstable"]' \
    //      'This is a nightly-only experimental API'
    // @count stability/struct.Unstable.html '//span[@class="stab unstable"]' 0
    pub foo: u32,
    pub bar: u32,
}
