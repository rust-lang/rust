#![warn(clippy::trailing_empty_array)]

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

#[must_use]
struct OnlyAnotherAttribute {
    field: i32,
    last: [usize; 0],
}

#[derive(Debug)]
struct OnlyADeriveAttribute {
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

const fn compute_zero_from_arg(x: usize) -> usize {
    x - 1
}
struct ZeroSizedWithConstFunction2 {
    field: i32,
    last: [usize; compute_zero_from_arg(1)],
}

struct ZeroSizedArrayWrapper([usize; 0]);

struct TupleStruct(i32, [usize; 0]);

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

struct NotLastField {
    f1: u32,
    zero_sized: [usize; 0],
    last: i32,
}

const ONE: usize = 1;
struct NonZeroSizedWithConst {
    field: i32,
    last: [usize; ONE],
}

#[derive(Debug)]
#[repr(C)]
struct AlsoADeriveAttribute {
    field: i32,
    last: [usize; 0],
}

#[must_use]
#[repr(C)]
struct AlsoAnotherAttribute {
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

#[repr(C)]
struct TupleStructReprC(i32, [usize; 0]);

type NamedTuple = (i32, [usize; 0]);

#[rustfmt::skip] // [rustfmt#4995](https://github.com/rust-lang/rustfmt/issues/4995)
struct ConstParamZeroDefault<const N: usize = 0> {
    field: i32,
    last: [usize; N],
}

struct ConstParamNoDefault<const N: usize> {
    field: i32,
    last: [usize; N],
}

#[rustfmt::skip] 
struct ConstParamNonZeroDefault<const N: usize = 1> {
    field: i32,
    last: [usize; N],
}

struct TwoGenericParams<T, const N: usize> {
    field: i32,
    last: [T; N],
}

type A = ConstParamZeroDefault;
type B = ConstParamZeroDefault<0>;
type C = ConstParamNoDefault<0>;
type D = ConstParamNonZeroDefault<0>;

fn main() {}
