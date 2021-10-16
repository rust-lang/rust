#![warn(clippy::trailing_zero_sized_array_without_repr_c)]
// #![feature(const_generics_defaults)] // see below

struct RarelyUseful {
    field: i32,
    last: [usize; 0],
}

#[repr(C)]
struct GoodReason {
    field: i32,
    last: [usize; 0],
}

struct OnlyFieldIsZeroSizeArray {
    first_and_last: [usize; 0],
}

struct GenericArrayType<T> {
    field: i32,
    last: [T; 0],
}

struct SizedArray {
    field: i32,
    last: [usize; 1],
}

const ZERO: usize = 0;
struct ZeroSizedFromExternalConst {
    field: i32,
    last: [usize; ZERO],
}

const ONE: usize = 1;
struct NonZeroSizedFromExternalConst {
    field: i32,
    last: [usize; ONE],
}

#[allow(clippy::eq_op)] // lmao im impressed
const fn compute_zero() -> usize {
    (4 + 6) - (2 * 5)
}
struct UsingFunction {
    field: i32,
    last: [usize; compute_zero()],
}

// NOTE: including these (along with the required feature) triggers an ICE. Should make sure the
// const generics people are aware of that if they weren't already.

// #[repr(C)]
// struct ConstParamOk<const N: usize = 0> {
//     field: i32,
//     last: [usize; N]
// }

// struct ConstParamLint<const N: usize = 0> {
//     field: i32,
//     last: [usize; N]
// }

// TODO: actually, uh,, no idea what behavior here would be
#[repr(packed)]
struct ReprPacked {
    small: u8,
    medium: i32,
    weird: [u64; 0],
}

// TODO: clarify expected behavior
#[repr(align(64))]
struct ReprAlign {
    field: i32,
    last: [usize; 0],
}

// TODO: clarify expected behavior
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

fn main() {}
