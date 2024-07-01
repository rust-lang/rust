//! We would like to try add `noundef` to small immediate arguments (the heuristic here is if the
//! immediate argument fits within target pointer width) where possible and legal. Adding `noundef`
//! attribute is legal iff the immediate argument do not have padding (indeterminate value
//! otherwise). Only simple immediates are considered currently (trivially no padding), but could be
//! potentially expanded to other small aggregate immediates that do not have padding (subject to
//! being able to correctly calculate "no padding").
//!
//! - We should recursively see through `#[repr(transparent)]` and `#[repr(Rust)]` layouts.
//! - Unions cannot have `noundef` because all unions are currently allowed to be `undef`. This
//!   property is "infectious", anything that contains unions also may not have `noundef` applied.

// ignore-tidy-linelength

#![crate_type = "lib"]

//@ compile-flags: -C no-prepopulate-passes

// We setup two revisions to check that `noundef` is only added when optimization is enabled.
//@ revisions: NoOpt Opt
//@ [NoOpt] compile-flags: -C opt-level=0
//@ [Opt] compile-flags: -O

// Presence of `noundef` depends on target pointer width (it's only applied when the immediate fits
// within target pointer width).
//@ only-64bit

// -------------------------------------------------------------------------------------------------

// # Positive test cases
//
// - Simple arrays of primitive types whose size fits within target pointer width (referred to as
//   "simple arrays" for the following positive test cases).
// - `#[repr(transparent)]` ADTs which eventually contain simple arrays.
// - `#[repr(Rust)]` ADTs which eventually contain simple arrays. This relies on rustc layout
//   behavior, and is not guaranteed by `#[repr(Rust)]`.

// ## Simple arrays

// NoOpt: define i64 @short_array_u64x1(i64 %{{.*}})
// Opt: define noundef i64 @short_array_u64x1(i64 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u64x1(v: [u64; 1]) -> [u64; 1] {
    v
}

// NoOpt: define i32 @short_array_u32x1(i32 %{{.*}})
// Opt: define noundef i32 @short_array_u32x1(i32 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u32x1(v: [u32; 1]) -> [u32; 1] {
    v
}

// NoOpt: define i64 @short_array_u32x2(i64 %{{.*}})
// Opt: define noundef i64 @short_array_u32x2(i64 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u32x2(v: [u32; 2]) -> [u32; 2] {
    v
}

// NoOpt: define i16 @short_array_u16x1(i16 %{{.*}})
// Opt: define noundef i16 @short_array_u16x1(i16 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u16x1(v: [u16; 1]) -> [u16; 1] {
    v
}

// NoOpt: define i32 @short_array_u16x2(i32 %{{.*}})
// Opt: define noundef i32 @short_array_u16x2(i32 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u16x2(v: [u16; 2]) -> [u16; 2] {
    v
}

// NoOpt: define i48 @short_array_u16x3(i48 %{{.*}})
// Opt: define noundef i48 @short_array_u16x3(i48 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u16x3(v: [u16; 3]) -> [u16; 3] {
    v
}

// NoOpt: define i64 @short_array_u16x4(i64 %{{.*}})
// Opt: define noundef i64 @short_array_u16x4(i64 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u16x4(v: [u16; 4]) -> [u16; 4] {
    v
}

// NoOpt: define i8 @short_array_u8x1(i8 %{{.*}})
// Opt: define noundef i8 @short_array_u8x1(i8 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u8x1(v: [u8; 1]) -> [u8; 1] {
    v
}

// NoOpt: define i16 @short_array_u8x2(i16 %{{.*}})
// Opt: define noundef i16 @short_array_u8x2(i16 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u8x2(v: [u8; 2]) -> [u8; 2] {
    v
}

// NoOpt: define i24 @short_array_u8x3(i24 %{{.*}})
// Opt: define noundef i24 @short_array_u8x3(i24 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u8x3(v: [u8; 3]) -> [u8; 3] {
    v
}

// NoOpt: define i64 @short_array_u8x8(i64 %{{.*}})
// Opt: define noundef i64 @short_array_u8x8(i64 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u8x8(v: [u8; 8]) -> [u8; 8] {
    v
}

// ## Small `#[repr(transparent)]` wrappers

#[repr(transparent)]
pub struct TransparentWrapper([u8; 4]);

// NoOpt: define i32 @repr_transparent_wrapper(i32 %{{.*}})
// Opt: define noundef i32 @repr_transparent_wrapper(i32 noundef %{{.*}})
#[no_mangle]
pub fn repr_transparent_wrapper(v: TransparentWrapper) -> TransparentWrapper {
    v
}

#[repr(transparent)]
pub struct RecursiveTransparentWrapper(TransparentWrapper);

// NoOpt: define i32 @recursive_repr_transparent_wrapper(i32 %{{.*}})
// Opt: define noundef i32 @recursive_repr_transparent_wrapper(i32 noundef %{{.*}})
#[no_mangle]
pub fn recursive_repr_transparent_wrapper(
    v: RecursiveTransparentWrapper,
) -> RecursiveTransparentWrapper {
    v
}

// ## Small `#[repr(Rust)]` wrappers
//
// Note that this relies on rustc self-consistency in handling simple `#[repr(Rust)]` wrappers, i.e.
// that `struct Foo([u8; 4])` has the same layout as its sole inner member and that no additional
// padding is introduced.

pub struct ReprRustWrapper([u8; 4]);

// NoOpt: define i32 @repr_rust_wrapper(i32 %{{.*}})
// Opt: define noundef i32 @repr_rust_wrapper(i32 noundef %{{.*}})
#[no_mangle]
pub fn repr_rust_wrapper(v: ReprRustWrapper) -> ReprRustWrapper {
    v
}

// ## Cases not handled
//
// - Aggregates that have no padding and fits within target pointer width which are not simple
//   arrays. Potentially aggregates such as tuples `(u32, u32)`. This is left as an exercise to the
//   reader (follow-up welcomed) :)

// No `noundef` annotation on return `{i32, i32}`, but argument does (when optimizations enabled).
// NoOpt: define { i32, i32 } @unhandled_small_pair_ret(i32 %v.0, i32 %v.1)
// Opt: define { i32, i32 } @unhandled_small_pair_ret(i32 noundef %v.0, i32 noundef %v.1)
#[no_mangle]
pub fn unhandled_small_pair_ret(v: (u32, u32)) -> (u32, u32) {
    v
}

// -------------------------------------------------------------------------------------------------

// # Negative test cases ()
//
// - Other representations (not `transparent` or `Rust`)
// - Unions cannot have `noundef` because they are allowed to be `undef`.
// - Array of unions still contains unions, so they cannot have `noundef`
// - Transparent unions are still unions, so they cannot have `noundef`

// ## Other representations

#[repr(C)]
pub struct ReprCWrapper([u8; 4]);

// NoOpt: define i32 @repr_c_immediate(i32 %0)
// Opt: define i32 @repr_c_immediate(i32 %0)
#[no_mangle]
pub fn repr_c_immediate(v: ReprCWrapper) -> ReprCWrapper {
    v
}

// ## Unions

union U {
    u1: u64,
    u2: [u8; 4],
}

// All unions can be `undef`, must not have `noundef` as immediate argument.
// NoOpt: define i64 @union_immediate(i64 %0)
// Opt: define i64 @union_immediate(i64 %0)
#[no_mangle]
pub fn union_immediate(v: U) -> U {
    v
}

// ## Array of unions
//
// Cannot have `noundef` because tainted by unions.

union SmallButDangerous {
    u1: [u8; 2],
    u2: u16,
}

// NoOpt: define i16 @one_elem_array_of_unions(i16 %0)
// Opt: define i16 @one_elem_array_of_unions(i16 %0)
#[no_mangle]
pub fn one_elem_array_of_unions(v: [SmallButDangerous; 1]) -> [SmallButDangerous; 1] {
    v
}

// NoOpt: define i32 @two_elem_array_of_unions(i32 %0)
// Opt: define i32 @two_elem_array_of_unions(i32 %0)
#[no_mangle]
pub fn two_elem_array_of_unions(v: [SmallButDangerous; 2]) -> [SmallButDangerous; 2] {
    v
}

// # `#[repr(transparent)]` unions

union Inner {
    i1: u8,
}

#[repr(transparent)]
pub struct TransparentUnionWrapper(Inner);

// NoOpt: define i8 @repr_transparent_union(i8 %v)
// Opt: define i8 @repr_transparent_union(i8 %v)
#[no_mangle]
pub fn repr_transparent_union(v: TransparentUnionWrapper) -> TransparentUnionWrapper {
    v
}
