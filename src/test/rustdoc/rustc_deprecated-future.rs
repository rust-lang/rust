#![feature(staged_api)]

#![stable(feature = "rustc_deprecated-future-test", since = "1.0.0")]

// @has rustc_deprecated_future/index.html '//*[@class="stab deprecated"]' \
//      'Deprecation planned'
// @has rustc_deprecated_future/struct.S1.html '//*[@class="stab deprecated"]' \
//      'Deprecating in 99.99.99: effectively never'
#[rustc_deprecated(since = "99.99.99", reason = "effectively never")]
#[stable(feature = "rustc_deprecated-future-test", since = "1.0.0")]
pub struct S1;

// @has rustc_deprecated_future/index.html '//*[@class="stab deprecated"]' \
//      'Deprecation planned'
// @has rustc_deprecated_future/struct.S2.html '//*[@class="stab deprecated"]' \
//      'Deprecating in a future Rust version: literally never'
#[rustc_deprecated(since = "TBD", reason = "literally never")]
#[stable(feature = "rustc_deprecated-future-test", since = "1.0.0")]
pub struct S2;
