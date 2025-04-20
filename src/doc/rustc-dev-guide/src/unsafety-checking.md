# Unsafety checking

Certain expressions in Rust can violate memory safety and as such need to be
inside an `unsafe` block or function. The compiler will also warn if an unsafe
block is used without any corresponding unsafe operations.

## Overview

The unsafety check is located in the [`check_unsafety`] module. It performs a
walk over the [THIR] of a function and all of its closures and inline constants.
It keeps track of the unsafe context: whether it has entered an `unsafe` block.
If an unsafe operation is used outside of an `unsafe` block, then an error is
reported. If an unsafe operation is used in an unsafe block then that block is
marked as used for [the unused_unsafe lint](#the-unused_unsafe-lint).

The unsafety check needs type information so could potentially be done on the
HIR, making use of typeck results, THIR or MIR. THIR is chosen because there are
fewer cases to consider than in HIR, for example unsafe function calls and
unsafe method calls have the same representation in THIR. The check is not done
on MIR because safety checks do not depend on control flow so MIR is not
necessary to use and MIR doesn't have as precise spans for some expressions.

Most unsafe operations can be identified by checking the `ExprKind` in THIR and
checking the type of the argument. For example, dereferences of a raw pointer
correspond to `ExprKind::Deref`s with an argument that has a raw pointer type.

Looking for unsafe Union field accesses is a bit more complex because writing to
a field of a union is safe. The checker tracks when it's visiting the left-hand
side of an assignment expression and allows union fields to directly appear
there, while erroring in all other cases. Union field accesses can also occur
in patterns, so those have to be walked as well.

The other complicated safety check is for writes to fields of layout constrained
structs (such as [`NonNull`]). These are found by looking for the borrow or
assignment expression and then visiting the subexpression being borrowed or
assigned with a separate visitor.

[THIR]: ./thir.md
[`check_unsafety`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir_build/check_unsafety/index.html
[`NonNull`]: https://doc.rust-lang.org/std/ptr/struct.NonNull.html

## The unused_unsafe lint

The unused_unsafe lint reports `unsafe` blocks that can be removed. The unsafety
checker records whenever it finds an operation that requires unsafe. The lint is
then reported if either:

- An `unsafe` block contains no unsafe operations
- An `unsafe` block is within another unsafe block, and the outer block
  isn't considered unused

```rust
#![deny(unused_unsafe)]
let y = 0;
let x: *const u8 = core::ptr::addr_of!(y);
unsafe { // lint reported for this block
    unsafe {
        let z = *x;
    }
    let safe_expr = 123;
}
unsafe {
    unsafe { // lint reported for this block
        let z = *x;
    }
    let unsafe_expr = *x;
}
```

## Other checks involving `unsafe`

[Unsafe traits] require an `unsafe impl` to be implemented, the check for this
is done as part of [coherence]. The `unsafe_code` lint is run as a lint pass on
the ast that searches for unsafe blocks, functions and implementations, as well
as certain unsafe attributes.

[Unsafe traits]: https://doc.rust-lang.org/reference/items/traits.html#unsafe-traits
[coherence]: https://github.com/rust-lang/rust/blob/master/compiler/rustc_hir_analysis/src/coherence/unsafety.rs
