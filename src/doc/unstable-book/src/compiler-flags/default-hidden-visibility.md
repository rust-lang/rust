# `default-hidden-visibility`

The tracking issue for this feature is: https://github.com/rust-lang/compiler-team/issues/656

------------------------

This flag can be used to override the target's
[`default_hidden_visibility`](https://doc.rust-lang.org/beta/nightly-rustc/rustc_target/spec/struct.TargetOptions.html#structfield.default_hidden_visibility)
setting.
Using `-Zdefault_hidden_visibility=yes` is roughly equivalent to Clang's
[`-fvisibility=hidden`](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fvisibility)
cmdline flag.
