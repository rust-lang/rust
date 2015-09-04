- Feature Name: raw-pointer-comparisons
- Start Date: 2015-05-27
- RFC PR: [rust-lang/rfcs#1135](https://github.com/rust-lang/rfcs/pull/1135)
- Rust Issue: [rust-lang/rust#28235](https://github.com/rust-lang/rust/issues/28236)

# Summary

Allow equality, but not order, comparisons between fat raw pointers
of the same type.

# Motivation

Currently, fat raw pointers can't be compared via either PartialEq or
PartialOrd (currently this causes an ICE). It seems to me that a primitive
type like a fat raw pointer should implement equality in some way.

However, there doesn't seem to be a sensible way to order raw fat pointers
unless we take vtable addresses into account, which is relatively weird.

# Detailed design

Implement PartialEq/Eq for fat raw pointers, defined as comparing both the
unsize-info and the address. This means that these are true:

```Rust
    &s as &fmt::Debug as *const _ == &s as &fmt::Debug as *const _ // of course
    &s.first_field as &fmt::Debug as *const _
        != &s as &fmt::Debug as *const _ // these are *different* (one
	                                 // prints only the first field,
					 // the other prints all fields).
```

But
```Rust
    &s.first_field as &fmt::Debug as *const _ as *const () ==
        &s as &fmt::Debug as *const _ as *const () // addresses are equal
```

# Drawbacks

Order comparisons may be useful for putting fat raw pointers into
ordering-based data structures (e.g. BinaryTree).

# Alternatives

@nrc suggested to implement heterogeneous comparisons between all thin
raw pointers and all fat raw pointers. I don't like this because equality
between fat raw pointers of different traits is false most of the
time (unless one of the traits is a supertrait of the other and/or the
only difference is in free lifetimes), and anyway you can always compare
by casting both pointers to a common type.

It is also possible to implement ordering too, either in unsize -> addr
lexicographic order or addr -> unsize lexicographic order.

# Unresolved questions

What form of ordering should be adopted, if any?


