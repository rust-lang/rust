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

struct GenericArrayType<T> {
    field: i32,
    last: [T; 0],
}

fn main() {}
