% The Unsafe Rust Programming Language

# NOTE: This is a draft document, and may contain serious errors

**This document is about advanced functionality and low-level development practices
in the Rust Programming Language. Most of the things discussed won't matter
to the average Rust programmer. However if you wish to correctly write unsafe
code in Rust, this text contains invaluable information.**

The Unsafe Rust Programming Language (TURPL) seeks to complement
[The Rust Programming Language Book][trpl] (TRPL).
Where TRPL introduces the language and teaches the basics, TURPL dives deep into
the specification of the language, and all the nasty bits necessary to write
Unsafe Rust. TURPL does not assume you have read TRPL, but does assume you know
the basics of the language and systems programming. We will not explain the
stack or heap. We will not explain the basic syntax.



[trpl]: https://doc.rust-lang.org/book/