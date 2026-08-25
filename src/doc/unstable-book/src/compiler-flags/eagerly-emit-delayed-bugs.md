# `eagerly-emit-delayed-bugs`

This feature is perma-unstable and has no tracking issue.

------------------------

This flag converts all [`span_delayed_bug()`] calls to [`bug!`] calls, exiting the compiler immediately and allowing you to generate a backtrace of where the delayed bug occurred.
For full documentation, see [the rustc-dev-guide][dev-guide-delayed].

[`bug!`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/macro.bug.html
[`span_delayed_bug()`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/struct.DiagCtxtHandle.html#method.span_delayed_bug
[dev-guide-delayed]: https://rustc-dev-guide.rust-lang.org/compiler-debugging.html#debugging-delayed-bugs
