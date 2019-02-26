#![feature(staged_api)]

#![stable(feature = "rustc_deprecated-future-test", since = "1.0.0")]

// @has rustc_deprecated_future/index.html '//*[@class="stab deprecated"]' \
//      'Deprecation planned'
// @has rustc_deprecated_future/struct.S.html '//*[@class="stab deprecated"]' \
//      'Deprecating in 99.99.99: effectively never'
#[rustc_deprecated(since = "99.99.99", reason = "effectively never")]
#[stable(feature = "rustc_deprecated-future-test", since = "1.0.0")]
pub struct S;
