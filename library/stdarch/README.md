stdsimd
=======

Experimental support for SIMD destined to eventually become part of Rust's
standard library

[![Build Status](https://travis-ci.org/rust-lang-nursery/stdsimd.svg?branch=master)](https://travis-ci.org/rust-lang-nursery/stdsimd)
[![Build status](https://ci.appveyor.com/api/projects/status/ix74qhmilpibn00x/branch/master?svg=true)](https://ci.appveyor.com/project/rust-lang-libs/stdsimd/branch/master)

This is a **work in progress**.

* [Documentation - i686][i686]
* [Documentation - x86\_64][x86_64]
* [Documentation - arm][arm]
* [Documentation - aarch64][aarch64]
* [How to get started][contrib]
* [How to help implement intrinsics][help-implement]

[contrib]: https://github.com/rust-lang-nursery/stdsimd/blob/master/CONTRIBUTING.md
[help-implement]: https://github.com/rust-lang-nursery/stdsimd/issues/40
[i686]: https://rust-lang-nursery.github.io/stdsimd/i686/stdsimd/
[x86_64]: https://rust-lang-nursery.github.io/stdsimd/x86_64/stdsimd/
[arm]: https://rust-lang-nursery.github.io/stdsimd/arm/stdsimd/
[aarch64]: https://rust-lang-nursery.github.io/stdsimd/aarch64/stdsimd/

### Approach

The main goal is to expose APIs defined by *vendors* with the least amount of
abstraction possible. On x86, for example, the API should correspond to that
provided by `emmintrin.h`.

# License

`stdsimd` is primarily distributed under the terms of both the MIT license and
the Apache License (Version 2.0), with portions covered by various BSD-like
licenses.

See LICENSE-APACHE, and LICENSE-MIT for details.
