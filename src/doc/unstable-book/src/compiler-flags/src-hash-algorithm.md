# `src-hash-algorithm`

The tracking issue for this feature is: [#70401](https://github.com/rust-lang/rust/issues/70401).

------------------------

The `-Z src-hash-algorithm` compiler flag controls which algorithm is used when hashing each source file. The hash is stored in the debug info and can be used by a debugger to verify the source code matches the executable.

Supported hash algorithms are: `md5`, `sha1`, and `sha256`. Note that not all hash algorithms are supported by all debug info formats.

By default, the compiler chooses the hash algorithm based on the target specification.
