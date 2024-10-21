# `default-visibility`

The tracking issue for this feature is: https://github.com/rust-lang/rust/issues/131090

------------------------

This flag can be used to override the target's
[`default_visibility`](https://doc.rust-lang.org/beta/nightly-rustc/rustc_target/spec/struct.TargetOptions.html#structfield.default_visibility)
setting.

This option only affects building of shared objects and should have no effect on executables.

Visibility an be set to one of three options:

* protected
* hidden
* interposable

## Hidden visibility

Using `-Zdefault-visibility=hidden` is roughly equivalent to Clang's
[`-fvisibility=hidden`](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fvisibility)
cmdline flag. Hidden symbols will not be exported from the created shared object, so cannot be
referenced from other shared objects or from executables.

## Protected visibility

Using `-Zdefault-visibility=protected` will cause rust-mangled symbols to be emitted with
"protected" visibility. This signals the compiler, the linker and the runtime linker that these
symbols cannot be overridden by the executable or by other shared objects earlier in the load order.

This will allow the compiler to emit direct references to symbols, which may improve performance. It
also removes the need for these symbols to be resolved when a shared object built with this option
is loaded.

Using protected visibility when linking with GNU ld prior to 2.40 will result in linker errors when
building for Linux. Other linkers such as LLD are not affected.

## Interposable

Using `-Zdefault-visibility=interposable` will cause symbols to be emitted with "default"
visibility. On platforms that support it, this makes it so that symbols can be interposed, which
means that they can be overridden by symbols with the same name from the executable or by other
shared objects earier in the load order.
