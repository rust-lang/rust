#![warn(clippy::trailing_zero_sized_array_without_repr_c)]
#![feature(const_generics_defaults)] // see below

// Do lint:

struct RarelyUseful {
    field: i32,
    last: [usize; 0],
}

struct OnlyField {
    first_and_last: [usize; 0],
}

struct GenericArrayType<T> {
    field: i32,
    last: [T; 0],
}

#[derive(Debug)]
struct OnlyAnotherAttributeDerive {
    field: i32,
    last: [usize; 0],
}

#[must_use]
struct OnlyAnotherAttributeMustUse {
    field: i32,
    last: [usize; 0],
}

const ZERO: usize = 0;
struct ZeroSizedWithConst {
    field: i32,
    last: [usize; ZERO],
}

#[allow(clippy::eq_op)]
const fn compute_zero() -> usize {
    (4 + 6) - (2 * 5)
}
struct ZeroSizedWithConstFunction {
    field: i32,
    last: [usize; compute_zero()],
}

struct LotsOfFields {
    f1: u32,
    f2: u32,
    f3: u32,
    f4: u32,
    f5: u32,
    f6: u32,
    f7: u32,
    f8: u32,
    f9: u32,
    f10: u32,
    f11: u32,
    f12: u32,
    f13: u32,
    f14: u32,
    f15: u32,
    f16: u32,
    last: [usize; 0],
}

// Don't lint

#[repr(C)]
struct GoodReason {
    field: i32,
    last: [usize; 0],
}

#[repr(C)]
struct OnlyFieldWithReprC {
    first_and_last: [usize; 0],
}

struct NonZeroSizedArray {
    field: i32,
    last: [usize; 1],
}

const ONE: usize = 1;
struct NonZeroSizedWithConst {
    field: i32,
    last: [usize; ONE],
}

#[derive(Debug)]
#[repr(C)]
struct OtherAttributesDerive {
    field: i32,
    last: [usize; 0],
}

#[must_use]
#[repr(C)]
struct OtherAttributesMustUse {
    field: i32,
    last: [usize; 0],
}

#[repr(packed)]
struct ReprPacked {
    field: i32,
    last: [usize; 0],
}

#[repr(C, packed)]
struct ReprCPacked {
    field: i32,
    last: [usize; 0],
}

#[repr(align(64))]
struct ReprAlign {
    field: i32,
    last: [usize; 0],
}
#[repr(C, align(64))]
struct ReprCAlign {
    field: i32,
    last: [usize; 0],
}

// NOTE: because of https://doc.rust-lang.org/stable/reference/type-layout.html#primitive-representation-of-enums-with-fields and I'm not sure when in the compilation pipeline that would happen
#[repr(C)]
enum DontLintAnonymousStructsFromDesuraging {
    A(u32),
    B(f32, [u64; 0]),
    C { x: u32, y: [u64; 0] },
}

// NOTE: including these (along with the required feature) triggers an ICE. Not sure why. Should
// make sure the const generics people are aware of that if they weren't already.

// #[repr(C)]
// struct ConstParamOk<const N: usize = 0> {
//     field: i32,
//     last: [usize; N]
// }

// struct ConstParamLint<const N: usize = 0> {
//     field: i32,
//     last: [usize; N]
// }

fn main() {
    let _ = OnlyAnotherAttributeMustUse { field: 0, last: [] };
    let _ = OtherAttributesMustUse { field: 0, last: [] };
}
