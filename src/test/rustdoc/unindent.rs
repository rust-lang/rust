#![feature(external_doc)]

#![crate_name = "foo"]

// @has foo/struct.Example.html
// @matches - '//pre[@class="rust rust-example-rendered"]' \
//     '(?m)let example = Example::new\(\)\n    \.first\(\)\n    \.second\(\)\n    \.build\(\);\Z'
/// ```rust
/// let example = Example::new()
///     .first()
#[cfg_attr(not(feature = "one"), doc = "    .second()")]
///     .build();
/// ```
pub struct Example;

// @has foo/struct.F.html
// @matches - '//pre[@class="rust rust-example-rendered"]' \
//     '(?m)let example = Example::new\(\)\n    \.first\(\)\n    \.another\(\)\n    \.build\(\);\Z'
///```rust
///let example = Example::new()
///    .first()
#[cfg_attr(not(feature = "one"), doc = "    .another()")]
///    .build();
/// ```
pub struct F;

// @has foo/struct.G.html
// @matches - '//pre[@class="rust rust-example-rendered"]' \
//     '(?m)let example = Example::new\(\)\n\.first\(\)\n    \.another\(\)\n\.build\(\);\Z'
///```rust
///let example = Example::new()
///.first()
#[cfg_attr(not(feature = "one"), doc = "    .another()")]
///.build();
///```
pub struct G;

// @has foo/struct.H.html
// @has - '//div[@class="docblock"]/p' 'no whitespace lol'
///no whitespace
#[doc = " lol"]
pub struct H;

// @has foo/struct.I.html
// @matches - '//pre[@class="rust rust-example-rendered"]' '(?m)4 whitespaces!\Z'
///     4 whitespaces!
#[doc = "something"]
pub struct I;

// @has foo/struct.J.html
// @matches - '//div[@class="docblock"]/p' '(?m)a\nno whitespace\nJust some text.\Z'
///a
///no whitespace
#[doc(include = "unindent.md")]
pub struct J;

// @has foo/struct.K.html
// @matches - '//pre[@class="rust rust-example-rendered"]' '(?m)4 whitespaces!\Z'
///a
///
///    4 whitespaces!
///
#[doc(include = "unindent.md")]
pub struct K;
