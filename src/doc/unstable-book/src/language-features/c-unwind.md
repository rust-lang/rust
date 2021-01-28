# `c_unwind`

The tracking issue for this feature is: [#74990]

[#74990]: https://github.com/rust-lang/rust/issues/74990

------------------------

Introduces four new ABI strings: "C-unwind", "stdcall-unwind",
"thiscall-unwind", and "system-unwind". These enable unwinding from other
languages (such as C++) into Rust frames and from Rust into other languages.

See [RFC 2945] for more information.

[RFC 2945]: https://github.com/rust-lang/rfcs/blob/master/text/2945-c-unwind-abi.md
