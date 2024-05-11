# `result_ffi_guarantees`

The tracking issue for this feature is: [#110503]

[#110503]: https://github.com/rust-lang/rust/issues/110503

------------------------

This feature adds the possibility of using `Result<T, E>` in FFI if T's niche
value can be used to describe E or vise-versa.

See [RFC 3391] for more information.

[RFC 3391]: https://github.com/rust-lang/rfcs/blob/master/text/3391-result_ffi_guarantees.md
