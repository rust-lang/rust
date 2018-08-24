#![feature(deprecated)]

// @has deprecated_future/struct.S.html '//*[@class="stab deprecated"]' \
//      'Deprecating in 99.99.99: effectively never'
#[deprecated(since = "99.99.99", note = "effectively never")]
pub struct S;
