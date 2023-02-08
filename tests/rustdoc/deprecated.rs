// @has deprecated/index.html '//*[@class="item-name"]/span[@class="stab deprecated"]' \
//      'Deprecated'
// @has - '//*[@class="desc docblock-short"]' 'Deprecated docs'

// @has deprecated/struct.S.html '//*[@class="stab deprecated"]' \
//      'Deprecated since 1.0.0: text'
/// Deprecated docs
#[deprecated(since = "1.0.0", note = "text")]
pub struct S;

// @matches deprecated/index.html '//*[@class="desc docblock-short"]' '^Docs'
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

// @matches deprecated/struct.X.html '//*[@class="stab deprecated"]' \
//      'Deprecated: shorthand reason$'
#[deprecated = "shorthand reason"]
pub struct X;
