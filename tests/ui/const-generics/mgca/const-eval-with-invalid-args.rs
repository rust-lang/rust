//! regression test for <https://github.com/rust-lang/rust/issues/139259>
#![expect(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(min_generic_const_args)]

// The previous ICE was an "invalid field access on immediate".
// If we remove `val: i32` from the field, another ICE occurs.
// "assertion `left == right` failed: invalid field type in
// Immediate::offset: scalar value has wrong size"
struct A {
    arr: usize,
    val: i32,
}

struct B<const N: A> {
    //~^ ERROR: `A` is forbidden as the type of a const generic parameter
    arr: [u8; N.arr],
    //~^ ERROR: complex const arguments must be placed inside of a `const` block
}

const C: u32 = 1;
fn main() {
    let b = B::<C> {arr: [1]};
    //~^ ERROR: the constant `C` is not of type `A`
    let _ = b.arr.len();
}
