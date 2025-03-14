//@ normalize-stderr: "pref: Align\([1-8] bytes\)" -> "pref: $$SOME_ALIGN"
//@ normalize-stderr: "randomization_seed: \d+" -> "randomization_seed: $$SEED"
#![feature(never_type, rustc_attrs, type_alias_impl_trait, repr_simd)]
#![crate_type = "lib"]

#[rustc_layout(debug)]
#[derive(Copy, Clone)]
enum E { Foo, Bar(!, i32, i32) } //~ ERROR: layout_of

#[rustc_layout(debug)]
struct S { f1: i32, f2: (), f3: i32 } //~ ERROR: layout_of

#[rustc_layout(debug)]
union U { f1: (i32, i32), f3: i32 } //~ ERROR: layout_of

#[rustc_layout(debug)]
type Test = Result<i32, i32>; //~ ERROR: layout_of

#[rustc_layout(debug)]
type T = impl std::fmt::Debug; //~ ERROR: layout_of
#[define_opaque(T)]
fn f() -> T {
    0i32
}

#[rustc_layout(debug)]
pub union V { //~ ERROR: layout_of
    a: [u16; 0],
    b: u8,
}

#[rustc_layout(debug)]
pub union W { //~ ERROR: layout_of
    b: u8,
    a: [u16; 0],
}

#[rustc_layout(debug)]
pub union Y { //~ ERROR: layout_of
    b: [u8; 0],
    a: [u16; 0],
}

#[rustc_layout(debug)]
#[repr(packed(1))]
union P1 { x: u32 } //~ ERROR: layout_of

#[rustc_layout(debug)]
#[repr(packed(1))]
union P2 { x: (u32, u32) } //~ ERROR: layout_of

#[repr(simd)]
#[derive(Copy, Clone)]
struct F32x4([f32; 4]);

#[rustc_layout(debug)]
#[repr(packed(1))]
union P3 { x: F32x4 } //~ ERROR: layout_of

#[rustc_layout(debug)]
#[repr(packed(1))]
union P4 { x: E } //~ ERROR: layout_of

#[rustc_layout(debug)]
#[repr(packed(1))]
union P5 { zst: [u16; 0], byte: u8 } //~ ERROR: layout_of

#[rustc_layout(debug)]
type X = std::mem::MaybeUninit<u8>; //~ ERROR: layout_of

#[rustc_layout(debug)]
const C: () = (); //~ ERROR: can only be applied to

impl S {
    #[rustc_layout(debug)]
    const C: () = (); //~ ERROR: can only be applied to
}

#[rustc_layout(debug)]
type Impossible = (str, str); //~ ERROR: cannot be known at compilation time

// Test that computing the layout of an empty union doesn't ICE.
#[rustc_layout(debug)]
union EmptyUnion {} //~ ERROR: has an unknown layout
//~^ ERROR: unions cannot have zero fields

// Test the error message of `LayoutError::TooGeneric`
// (this error is never emitted to users).
#[rustc_layout(debug)]
type TooGeneric<T> = T; //~ ERROR: does not have a fixed layout
