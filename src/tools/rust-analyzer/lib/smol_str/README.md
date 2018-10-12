# smol_str

[![Build Status](https://travis-ci.org/matklad/smol_str.svg?branch=master)](https://travis-ci.org/matklad/smol_str)
[![Crates.io](https://img.shields.io/crates/v/smol_str.svg)](https://crates.io/crates/smol_str)
[![API reference](https://docs.rs/smol_str/badge.svg)](https://docs.rs/smol_str/)


A `SmolStr` is a string type that has the following properties:

* `size_of::<SmolStr>() == size_of::<String>()`
* `Clone` is `O(1)`
* Strings are stack-allocated if they are:
    * Up to 22 bytes long
    * Longer than 22 bytes, but substrings of `WS` (see `src/lib.rs`). Such strings consist
    solely of consecutive newlines, followed by consecutive spaces
* If a string does not satisfy the aforementioned conditions, it is heap-allocated

Unlike `String`, however, `SmolStr` is immutable. The primary use case for
`SmolStr` is a good enough default storage for tokens of typical programming
languages. Strings consisting of a series of newlines, followed by a series of
whitespace are a typical pattern in computer programs because of indentation.
Note that a specialized interner might be a better solution for some use cases.
