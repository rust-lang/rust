//! Test for <https://github.com/rust-lang/rust/issues/151202>: we shouldn't expect that the operand
//! of `!` or `-` has the same type as the overall result of the operation, since `Not` and `Neg`
//! impls can have differing operand and output types.
//@ check-pass

fn main() {
    // If we propagated the expected type of `i8` here to the `{&0i8}` operands, the blocks would
    // force a coercion from `&i8` to `i8`, resulting in a type error.
    let _: i8 = -{&0i8};
    let _: i8 = !{&0i8};
}
