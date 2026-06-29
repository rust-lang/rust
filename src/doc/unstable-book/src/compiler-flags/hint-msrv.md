# `hint-msrv`

The tracking issue for this feature is: [#157574](https://github.com/rust-lang/rust/issues/157574).

------------------------

This feature allows you to specify a minimum Rust version for the crate, which will affect lint
emission. If following a lint suggestion would raise the MSRV above the provided value, it should
not be emitted.
