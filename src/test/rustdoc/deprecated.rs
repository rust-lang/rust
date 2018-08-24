#![feature(deprecated)]

// @has deprecated/struct.S.html '//*[@class="stab deprecated"]' \
//      'Deprecated since 1.0.0: text'
#[deprecated(since = "1.0.0", note = "text")]
pub struct S;
