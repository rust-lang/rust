# typed_key

[![Build Status](https://travis-ci.org/matklad/smol_str.svg?branch=master)](https://travis-ci.org/matklad/smol_str)
[![Crates.io](https://img.shields.io/crates/v/smol_str.svg)](https://crates.io/crates/smol_str)
[![API reference](https://docs.rs/smol_str/badge.svg)](https://docs.rs/smol_str/)


A `SmolStr` is a string type that has the following properties

  * `size_of::<SmolStr>() == size_of::<String>()`
  * Strings up to 22 bytes long do not use heap allocations
  * Runs of `\n` and space symbols (typical whitespace pattern of indentation
    in programming laguages) do not use heap allocations
  * `Clone` is `O(1)`

Unlike `String`, however, `SmolStr` is immutable. The primary use-case for
`SmolStr` is a good enough default storage for tokens of typical programming
languages. A specialized interner might be a better solution for some use-cases.

Intenrally, `SmolStr` is roughly an `enum { Heap<Arc<str>>, Inline([u8; 22]) }`.
