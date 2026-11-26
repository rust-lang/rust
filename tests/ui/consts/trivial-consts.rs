// This test ensures that `trivial_const` is correctly detecting trivial consts. It relies on a
// useful fact: the MIR pretty printer prints trivial consts in a simpler form in order to avoid
// calling `mir_for_ctfe`. E.g. for `const X: u8 = 0` (trivial) it prints:
//
//   const X: u8 = const 0_u8;
//
// but for `const X: u8 = 0|0` (non-trivial) it prints a full MIR body
//
//   const X: u8 = {
//       let mut _0: u8;
//
//       bb0: {
//           _0 = BitOr(const 0_u8, const 0_u8);
//           return;
//       }
//   }
//
//@ check-pass
//@ compile-flags: -Zunpretty=mir
//@ edition: 2024
//@ needs-unwind (because the output includes unwinding details)

#![allow(unused_braces)]
#![allow(unused)]
#![feature(type_alias_impl_trait)]

use std::fmt::Debug;
use std::time::Duration;

//---------------------------------------------------------------------------
// Trivial consts
//---------------------------------------------------------------------------

// Literals, with some negations
const A1: u8 = 0;
const A2: u8 = (99u8);
const A3: i8 = -128;
const A4: u32 = 0xffff_ffff;
const A5: i32 = 0x7fff_ffff_i32;
const A6: u128 = 123;
const A7: i128 = -170141183460469231731687303715884105728;
const A8: f32 = (-(0.0));
const A9: f64 = 12.3e8_f64;
const A10: bool = true;
const A11: char = 'x';
const A12: &[u8; 3] = b"foo";
const A13: &str = "foo";
const A14: &std::ffi::CStr = c"foo";

// Coercion casts
const B1: usize = 0 as usize;
const B2: f32 = 99.9 as f32 as f32 as f32;

// Blocks, including some item statements, and tail-less forms
const C1: usize = { 0 };
const C2: usize = {{ 0 as usize }};
const C3: isize = {{{{{{ -20 }}}}}};
const C4: usize = { struct S; 100 };
const C5: () = {};
const C6: () = {
    struct S;
    impl S {}
    fn _f() {}
};
const C7: u32 = {
    let _: usize;
    0
};
const C8: u8 = unsafe { 0 };
const C9: usize = 'a: { 0 };

// Non-`const` cases
enum E { V1 = 5, V2 = 6 }   // 5 and 6 are the trivial consts
static ARR: [u8; 3] = [0; 3]; // 3 is a trivial const twice

// Paths
const D1: usize = C1;
const D2: usize = (D1);
const D3: usize = { D2 };
const D4: f32 = std::f32::consts::PI;
const D5: char = char::MAX;
struct S;
impl S {
    const K1: usize = 1;
    const K2: usize = Self::K1;
    const K3: Self = Self;
}
const D6: usize = S::K1;
const D7: usize = { ({ S::K2 }) as usize };

//---------------------------------------------------------------------------
// Non-trivial consts
//---------------------------------------------------------------------------

// RHS is not a literal, negated literal, or path to a trivial const
const E1: usize = 0|0;
const E2: bool = 1 < 2;
const E3: u32 = !0;
const E4: u32 = 1 << 4;
const E5: i8 = -(-1);
const E6: i32 = -A5;
const E7: i8 = E5;
static STATIC: u16 = 0;
const E8: u16 = STATIC;
const E9: S = S;                // constructor, not const
const E10: Option<u32> = None;  // constructor, not const
const E11: Duration = Duration::ZERO; // a function call: `Duration::from_nanos(0)`

// Casts
const F1: u8 = '\n' as u8;
const F2: u8 = true as u8;
const F3: *const u8 = 23 as *const u8;
const F4: &str = "blah" as &'static str; // coercion cast with user-provided type

// Adjustments
const G1: &[u8] = b"foo";
const G2: &[u8] = b"bar" as &[u8];
const G3: &str = "foo" as &str;

// Blocks
const H1: usize = { 1; 2 };     // expression statement
const H2: () = { (); };         // not quite tail-less
const H3: () = { 0; };          // not quite tail-less
const H4: u32 = {
    let _ = 3;                  // non-item statement
    0
};

// Inline const
fn inline_const() -> usize {
    const { 0 }
}

// TAIT
pub type ImplDebug = impl Debug;
#[define_opaque(ImplDebug)]
const I1: ImplDebug = 5;

// Generic args
trait Tr {
    const A: usize;
    const B: usize = Self::A;   // `Self` is a generic arg
}
impl Tr for S {
    const A: usize = 0;         // (this one is trivial)
}
struct W<T> { t: T }
impl<T: Tr> W<T> {
    const C: usize = T::A;      // `T` is a generic arg
}
impl<T> W<T> {
    const K: usize = 0;         // (this one is trivial)
}
const J1: usize = <S as Tr>::A; // `S` is a generic arg
const J2: usize = S::A;         // `S` is a generic arg
const J3: usize = W::<u8>::K;   // `u8` is a generic arg

fn main() {}
