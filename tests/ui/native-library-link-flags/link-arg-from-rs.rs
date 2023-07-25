// link-arg is not supposed to be usable in #[link] attributes

//@compile-flags:
// ignore-tidy-linelength
//@error-pattern: error[E0458]: unknown link kind `link-arg`, expected one of: static, dylib, framework, raw-dylib

#[link(kind = "link-arg")]
extern "C" {}
pub fn main() {}
