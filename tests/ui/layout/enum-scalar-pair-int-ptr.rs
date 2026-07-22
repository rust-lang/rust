//@ normalize-stderr: "pref: Align\([1-8] bytes\)" -> "pref: $$PREF_ALIGN"
//@ normalize-stderr: "u[0-9]+ is" -> "u?? is"
//@ normalize-stderr: "pointer is 0..=[0-9]+" -> "pointer is $$VALID_RANGE"
//@ normalize-stderr: "b_offset: Size\([0-9]+ bytes\)" -> "b_offset: Size(? bytes)"

//! Enum layout tests related to scalar pairs with an int/ptr common primitive.

#![feature(rustc_attrs)]
#![feature(never_type)]
#![crate_type = "lib"]

#[rustc_dump_layout(backend_repr)]
enum ScalarPairPointerWithInt { //~ERROR: backend_repr: ScalarPair
    A(usize),
    B(Box<()>),
}

// Negative test--ensure that pointers are not commoned with integers
// of a different size. (Assumes that no target has 8 bit pointers, which
// feels pretty safe.)
#[rustc_dump_layout(backend_repr)]
enum NotScalarPairPointerWithSmallerInt { //~ERROR: backend_repr: Memory
    A(u8),
    B(Box<()>),
}
