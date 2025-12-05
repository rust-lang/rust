//@ normalize-stderr: "at byte `\d+`" -> "at byte `$$BYTE`"
//@ normalize-stderr: "`[^`\n]*invalid-utf8-binary-file\.bin`" -> "`$DIR/invalid-utf8-binary-file.bin`"
//@ rustc-env:INVALID_UTF8_BIN={{src-base}}/include-macros/invalid-utf8-binary-file.bin

//! Ensure that ICE does not occur when reading an invalid UTF8 file with an absolute path.
//! regression test for issue <https://github.com/rust-lang/rust/issues/149304>

#![doc = include_str!(concat!(env!("INVALID_UTF8_BIN")))]
//~^ ERROR: wasn't a utf-8 file
//~| ERROR: attribute value must be a literal

fn main() {}
