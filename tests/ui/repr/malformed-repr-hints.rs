// Regression test for various ICEs inspired by
// https://github.com/rust-lang/rust/issues/83921#issuecomment-814640734

//@ compile-flags: -Zdeduplicate-diagnostics=yes

#[repr(packed())]
//~^ ERROR: incorrect `repr(packed)` attribute format
struct S1;

#[repr(align)]
//~^ ERROR: invalid `repr(align)` attribute
struct S2;

#[repr(align(2, 4))]
//~^ ERROR: incorrect `repr(align)` attribute format
struct S3;

#[repr(align())]
//~^ ERROR: incorrect `repr(align)` attribute format
struct S4;

// Regression test for issue #118334:
#[repr(Rust(u8))]
//~^ ERROR: invalid representation hint
#[repr(Rust(0))]
//~^ ERROR: invalid representation hint
#[repr(Rust = 0)]
//~^ ERROR: invalid representation hint
struct S5;

#[repr(i8())]
//~^ ERROR: invalid representation hint
enum E1 { A, B }

#[repr(u32(42))]
//~^ ERROR: invalid representation hint
enum E2 { A, B }

#[repr(i64 = 2)]
//~^ ERROR: invalid representation hint
enum E3 { A, B }

fn main() {}
