#![feature(deprecated)]

// @matches deprecated/index.html '//*[@class="docblock-short"]' \
//      '^\[Deprecated\] Deprecated docs'
// @has deprecated/struct.S.html '//*[@class="stab deprecated"]' \
//      'Deprecated since 1.0.0: text'
/// Deprecated docs
#[deprecated(since = "1.0.0", note = "text")]
pub struct S;

// @matches deprecated/index.html '//*[@class="docblock-short"]' '^Docs'
/// Docs
pub struct T;

// @matches deprecated/struct.U.html '//*[@class="stab deprecated"]' \
//      'Deprecated since 1.0.0$'
#[deprecated(since = "1.0.0")]
pub struct U;

// @matches deprecated/struct.V.html '//*[@class="stab deprecated"]' \
//      'Deprecated: text$'
#[deprecated(note = "text")]
pub struct V;

// @matches deprecated/struct.W.html '//*[@class="stab deprecated"]' \
//      'Deprecated$'
#[deprecated]
pub struct W;
