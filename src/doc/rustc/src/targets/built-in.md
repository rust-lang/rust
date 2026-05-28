# Built-in Targets

`rustc` ships with the ability to compile to many targets automatically, we
call these "built-in" targets, and they generally correspond to targets that
the team is supporting directly. To see the list of built-in targets, you can
run `rustc --print target-list`.

Typically, a target needs a compiled copy of the Rust standard library to
work. If using [rustup], then check out the documentation on
[Cross-compilation][rustup-cross] on how to download a pre-built standard
library built by the official Rust distributions. Most targets will need a
system linker, and possibly other things.

[rustup]: https://github.com/rust-lang/rustup
[rustup-cross]: https://rust-lang.github.io/rustup/cross-compilation.html
