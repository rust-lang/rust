# `treat-err-as-bug`

This feature is perma-unstable and has no tracking issue.

------------------------

This flag converts the selected error to a [`bug!`] call, exiting the compiler immediately and allowing you to generate a backtrace of where the error occurred.
For full documentation, see [the rustc-dev-guide][dev-guide-backtrace].

Note that the compiler automatically sets `RUST_BACKTRACE=1` for itself, and so you do not need to set it yourself when using this flag.

[`bug!`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/macro.bug.html
[dev-guide-backtrace]: https://rustc-dev-guide.rust-lang.org/compiler-debugging.html#getting-a-backtrace-for-errors
