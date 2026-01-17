//! Test for #151202: we shouldn't expect that the operand of `!` or `-` has the same type as the
//! overall result of the operation. That expectation was sometimes enforced, leading to errors on
//! uses of `Not` and `Neg` impls where the operand and result types don't match. In particular,
//! when the operand is a block, setting an expected type for it means we require the block's result
//! to coerce to that expected type.

fn main() {
    let _: i8 = -{&0i8};
    //~^ ERROR mismatched types
    let _: i8 = !{&0i8};
    //~^ ERROR mismatched types
}
