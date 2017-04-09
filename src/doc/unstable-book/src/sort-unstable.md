# `sort_unstable`

The tracking issue for this feature is: [#40585]

[#40585]: https://github.com/rust-lang/rust/issues/40585

------------------------

The default `sort` method on slices is stable. In other words, it guarantees
that the original order of equal elements is preserved after sorting. The
method has several undesirable characteristics:

1. It allocates a sizable chunk of memory.
2. If you don't need stability, it is not as performant as it could be.

An alternative is the new `sort_unstable` feature, which includes these
methods for sorting slices:

1. `sort_unstable`
2. `sort_unstable_by`
3. `sort_unstable_by_key`

Unstable sorting is generally faster and makes no allocations. The majority
of real-world sorting needs doesn't require stability, so these methods can
very often come in handy.

Another important difference is that `sort` lives in `libstd` and
`sort_unstable` lives in `libcore`. The reason is that the former makes
allocations and the latter doesn't.

A simple example:

```rust
#![feature(sort_unstable)]

let mut v = [-5, 4, 1, -3, 2];

v.sort_unstable();
assert!(v == [-5, -3, 1, 2, 4]);
```
