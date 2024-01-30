# `mixed_utf8_literals`

The tracking issue for this feature is: [#116907]

[#116907]: https://github.com/rust-lang/rust/issues/116907

------------------------

This feature extends the syntax of string literals in the following ways.
- Byte string literals can contain unicode chars (e.g. `b"🦀"`) and unicode
  escapes (e.g. `b"\u{1f980}"`.
- Raw byte string literals can contain unicode chars (e.g. `br"🦀"`).

This makes it easier to work with strings that are mostly UTF-8 encoded but
also contain some non UTF-8 bytes, which are sometimes called "conventionally
UTF-8" strings.
