# `rust-lang/rust` Licenses

The `rustc` compiler source and standard library are dual licensed under the [Apache License v2.0](https://github.com/rust-lang/rust/blob/master/LICENSE-APACHE) and the [MIT License](https://github.com/rust-lang/rust/blob/master/LICENSE-MIT) unless otherwise specified.

Detailed licensing information is available in the [COPYRIGHT document](https://github.com/rust-lang/rust/blob/master/COPYRIGHT) of the `rust-lang/rust` repository.

## Guidelines for reviewers

In general, reviewers need to be looking not only for the code quality of contributions but also
that they are properly licensed.
We have some tips below for things to look out for when reviewing, but if you ever feel uncertain
as to whether some code might be properly licensed, err on the safe side — reach out to the Council
or Compiler Team Leads for feedback!

Things to watch out for:

- The PR author states that they copied, ported, or adapted the code from some other source.
- There is a comment in the code pointing to a webpage or describing where the algorithm was taken
from.
- The algorithm or code pattern seems like it was likely copied from somewhere else.
- When adding new dependencies, double check the dependency's license.

In all of these cases, we will want to check that source to make sure it is licensed in a way
that is compatible with Rust’s license.

Examples

- Porting C code from a GPL project, like GNU binutils, is not allowed. That would require Rust
itself to be licensed under the GPL.
- Copying code from an algorithms text book may be allowed, but some algorithms are patented.

## Porting

Contributions to rustc, especially around platform and compiler intrinsics, often include porting
over work from other projects, mainly LLVM and GCC.

Some general rules apply:

- Copying work needs to adhere to the original license
    - This applies to direct copy & paste
    - This also applies to code you looked at and ported

In general, taking inspiration from other codebases is fine, but please exercise caution when
porting code.

Ports of full libraries (e.g. C libraries shipped with LLVM) must keep the license of the original
library.
