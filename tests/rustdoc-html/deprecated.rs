//@ has deprecated/index.html '//dt/span[@class="stab deprecated"]' 'Deprecated'
//@ has - '//dd' 'Deprecated docs'

//@ has deprecated/type.S.html '//*[@class="stab deprecated"]' \
//      'Deprecated since 1.0.0: text'
/// Deprecated docs
#[deprecated(since = "1.0.0", note = "text")]
pub struct S;

//@ matches deprecated/index.html '//dd' '^Docs'
/// Docs
pub struct T;

//@ matches deprecated/type.U.html '//*[@class="stab deprecated"]' \
//      'Deprecated since 1.0.0$'
#[deprecated(since = "1.0.0")]
pub struct U;

//@ matches deprecated/type.V.html '//*[@class="stab deprecated"]' \
//      'Deprecated: text$'
#[deprecated(note = "text")]
pub struct V;

//@ matches deprecated/type.W.html '//*[@class="stab deprecated"]' \
//      'Deprecated$'
#[deprecated]
pub struct W;

//@ matches deprecated/type.X.html '//*[@class="stab deprecated"]' \
//      'Deprecated: shorthand reason: code$'
#[deprecated = "shorthand reason: `code`"]
pub struct X;

//@ matches deprecated/type.Y.html '//*[@class="stab deprecated"]//p[1]' 'multiple'
//@ matches deprecated/type.Y.html '//*[@class="stab deprecated"]//p[2]' 'paragraphs'
#[deprecated = "multiple\n\nparagraphs"]
pub struct Y;
