#![feature(deprecated)]

// @has deprecated_future/index.html '//*[@class="stab deprecated"]' \
//      'Deprecated'
// @has deprecated_future/struct.S.html '//*[@class="stab deprecated"]' \
//      'Deprecated since 99.99.99: effectively never'
#[deprecated(since = "99.99.99", note = "effectively never")]
pub struct S;
