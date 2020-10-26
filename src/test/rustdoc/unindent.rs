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
