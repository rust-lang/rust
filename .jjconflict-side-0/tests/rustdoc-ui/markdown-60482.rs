// This code caused a panic in `pulldown-cmark` 0.4.1.
// https://github.com/rust-lang/rust/issues/60482

//@ check-pass

pub const BASIC_UNICODE: bool = true;

/// # `BASIC_UNICODE`: `A` `|`
/// ```text
/// ```
pub const BASIC_FONTS: bool = true;
