# generic_const_exprs

Allows non-trivial generic constants which have to be shown to successfully evaluate
to a value by being part of an item signature.

The tracking issue for this feature is: [#76560]


[#76560]: https://github.com/rust-lang/rust/issues/76560

------------------------

Warning: This feature is incomplete; its design and syntax may change.

See also: [min_generic_const_args], [generic_const_args]

[min_generic_const_args]: min-generic-const-args.md
[generic_const_args]: generic-const-args.md

## Examples

```rust
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

// Use parameters that depend on a generic argument.
struct Foo<const N: usize>
where
    [(); N + 1]:,
{
    array: [usize; N + 1],
}

// Use generic parameters in const operations.
trait Bar {
    const X: usize;
    const Y: usize;
}

// Note `B::X * B::Y`.
const fn baz<B: Bar>(x: [usize; B::X], y: [usize; B::Y]) -> [usize; B::X * B::Y] {
    let mut out = [0; B::X * B::Y];
    let mut i = 0;
    while i < B::Y {
        let mut j = 0;
        while j < B::X {
            out[i * B::X + j] = y[i].saturating_mul(x[j]);
            j += 1;
        }
        i += 1;
    }
    out
}


// Create a new type based on a generic argument.
pub struct Grow<const N: usize> {
    arr: [usize; N],
}

impl<const N: usize> Grow<N> {
    pub const fn grow(self, val: usize) -> Grow<{ N + 1 }> {
        let mut new_arr = [0; { N + 1 }];
        let mut idx = 0;
        while idx < N {
            new_arr[idx] = self.arr[idx];
            idx += 1;
        }
        new_arr[N] = val;
        Grow { arr: new_arr }
    }
}
```
