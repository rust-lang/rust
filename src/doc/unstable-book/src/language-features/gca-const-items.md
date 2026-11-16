# gca_const_items

Allows using generics in more complex const expressions, based on definitional equality.

The tracking issue for this feature is: [#151972]

[#151972]: https://github.com/rust-lang/rust/issues/151972

------------------------

Warning: This feature is incomplete; its design and syntax may change.

This feature enables many of the same use cases supported by [generic_const_exprs],
but based on the machinery developed for [gca_min_const_items]. In a way, it is
meant to be an interim successor for GCE (though it might not currently support all
the valid cases that supported by GCE).

See also: [generic_const_items]

[gca_min_const_items]: gca-min-const-items.md
[generic_const_exprs]: generic-const-exprs.md
[generic_const_items]: generic-const-items.md

## Examples

<!-- NOTE(ignore) gca_const_items requires -Znext-solver to compile -->
```rust,ignore (requires-Z-next-solver)
#![feature(generic_const_items)]
#![feature(gca_min_const_items)]
#![feature(gca_const_items)]
#![expect(incomplete_features)]

type const ADD1<const N: usize>: usize = const { N + 1 };

type const INC<const N: usize>: usize = ADD1::<N>;

const ARR: [(); ADD1::<0>] = [(); INC::<0>];
```
