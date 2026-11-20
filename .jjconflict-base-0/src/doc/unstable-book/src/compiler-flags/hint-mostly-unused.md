# `hint-mostly-unused`

This flag hints to the compiler that most of the crate will probably go unused.
The compiler can optimize its operation based on this assumption, in order to
compile faster. This is a hint, and does not guarantee any particular behavior.

This option can substantially speed up compilation if applied to a large
dependency where the majority of the dependency does not get used. This flag
may slow down compilation in other cases.

Currently, this option makes the compiler defer as much code generation as
possible from functions in the crate, until later crates invoke those
functions. Functions that never get invoked will never have code generated for
them. For instance, if a crate provides thousands of functions, but only a few
of them will get called, this flag will result in the compiler only doing code
generation for the called functions. (This uses the same mechanisms as
cross-crate inlining of functions.) This does not affect `extern` functions, or
functions marked as `#[inline(never)]`.

To try applying this flag to one dependency out of a dependency tree, use the
[`profile-rustflags`](https://doc.rust-lang.org/cargo/reference/unstable.html#profile-rustflags-option)
feature of nightly cargo:

```toml
cargo-features = ["profile-rustflags"]

# ...
[dependencies]
mostly-unused-dependency = "1.2.3"

[profile.release.package.mostly-unused-dependency]
rustflags = ["-Zhint-mostly-unused"]
```
