#![warn(clippy::trailing_zero_sized_array_without_repr_c)]

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

// struct GenericArrayType<T> {
//     field: i32,
//     last: [T; 0],
// }

// struct SizedArray {
//     field: i32,
//     last: [usize; 1],
// }

// const ZERO: usize = 0;
// struct ZeroSizedFromExternalConst {
//     field: i32,
//     last: [usize; ZERO],
// }

// const ONE: usize = 1;
// struct NonZeroSizedFromExternalConst {
//     field: i32,
//     last: [usize; ONE],
// }

// #[allow(clippy::eq_op)] // lmao im impressed
// const fn compute_zero() -> usize {
//     (4 + 6) - (2 * 5)
// }
// struct UsingFunction {
//     field: i32,
//     last: [usize; compute_zero()],
// }

// // TODO: same
// #[repr(packed)]
// struct ReprPacked {
//     small: u8,
//     medium: i32,
//     weird: [u64; 0],
// }

// // TODO: actually, uh,, 
// #[repr(align(64))]
// struct ReprAlign {
//     field: i32,
//     last: [usize; 0],
// }
// #[repr(C, align(64))]
// struct ReprCAlign {
//     field: i32,
//     last: [usize; 0],
// }

// #[repr(C)]
// enum DontLintAnonymousStructsFromDesuraging {
//     A(u32),
//     B(f32, [u64; 0]),
//     C { x: u32, y: [u64; 0] },
// }

fn main() {}
