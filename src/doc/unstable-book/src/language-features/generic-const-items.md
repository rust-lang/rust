# generic_const_items

Allows generic parameters and where-clauses on free & associated const items.

The tracking issue for this feature is: [#113521]

[#113521]: https://github.com/rust-lang/rust/issues/113521

------------------------

Warning: This feature is an [experiment] and lacks an RFC.
There are no guarantees that it will ever be stabilized.

See also: [generic_const_exprs], [min_generic_const_args].

[experiment]: https://lang-team.rust-lang.org/how_to/experiment.html
[generic_const_exprs]: generic-const-exprs.md
[min_generic_const_args]: min-generic-const-args.md

## Examples

### Generic constant values

```rust
#![allow(incomplete_features)]
#![feature(generic_const_items)]

const GENERIC_VAL<const ARG: usize>: usize = ARG + 1;

#[test]
fn generic_const_arg() {
    assert_eq!(GENERIC_VAL::<1>, 2);
    assert_eq!(GENERIC_VAL::<2>, 3);
}
```

### Conditional constants

```rust
#![allow(incomplete_features)]
#![feature(generic_const_items)]

// `GENERIC_VAL::<0>` will fail to compile
const GENERIC_VAL<const ARG: usize>: usize = if ARG > 0 { ARG + 1 } else { panic!("0 value") };

// Will fail to compile if the `Copy` derive is removed.
const COPY_MARKER<C: Copy>: () = ();

#[derive(Clone, Copy)]
struct Foo;

const FOO_IS_COPY: () = COPY_MARKER::<Foo>;
```
