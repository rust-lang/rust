# `c_unwind`

The tracking issue for this feature is: [#74990]

[#74990]: https://github.com/rust-lang/rust/issues/74990

------------------------

Introduces new ABI strings:
- "C-unwind"
- "cdecl-unwind"
- "stdcall-unwind"
- "fastcall-unwind"
- "vectorcall-unwind"
- "thiscall-unwind"
- "aapcs-unwind"
- "win64-unwind"
- "sysv64-unwind"
- "system-unwind"

These enable unwinding from other languages (such as C++) into Rust frames and
from Rust into other languages.

See [RFC 2945] for more information.

[RFC 2945]: https://github.com/rust-lang/rfcs/blob/master/text/2945-c-unwind-abi.md
