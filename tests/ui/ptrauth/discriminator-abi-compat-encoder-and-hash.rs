// Exercises the `fn` ABI-compatibility guarantees documented at
// https://doc.rust-lang.org/std/primitive.fn.html#abi-compatibility
// in the context of function pointer type discrimination.
//
// Within each group, every function's parameter type must be ABI-compatible
// with every other's, and must produce an identical discriminator. `*_neg`
// functions are negative controls and must NOT match the group they sit beside.
//
// NOTE: The doc's fourth compatibility rule
// "Any two fn (function pointer) types are ABI-compatible with each other if
// they have the same ABI string or the ABI string only differs in a trailing
// -unwind, independent of the rest of their signature. (This means you can pass
// fn() to a function expecting fn(i32), and the call will be valid ABI-wise.
// The callee receives the result of transmuting the function pointer from fn()
// to fn(i32); that transmutation is itself a well-defined operation, it’s just
// almost certainly UB to later call that function pointer.)" is
// satisfied for function pointer types used as VALUES (e.g. a callback parameter).
// Those collapse to 'P' via the same blanket pointer-merge that handles
// every other pointer-like type. It is deliberately *NOT HONORED* for the top-level signature
// being authenticated here.
//
// Honoring the rule fully would mean that every extern "C"/"System" function
// produces the identical discriminator, making function pointer type
// discrimination useless.

//@ dont-require-annotations: ERROR
#![crate_type = "lib"]
#![feature(rustc_attrs)]
#![allow(dead_code)]
#![allow(improper_ctypes_definitions)]
#![feature(allocator_ext)]

use std::alloc::{AllocError, Allocator, Global, Layout};
use std::marker::PhantomData;
use std::num::NonZero;
use std::ptr::NonNull;

struct SomeStruct {
    _x: i32,
}

// "*const T, *mut T, &T, &mut T, Box<T> (specifically, only Box<T, Global>),
// and NonNull<T> are all ABI-compatible with each other for all T. They are
// also ABI-compatible with each other for different T if they have the same
// metadata type (<T as Pointee>::Metadata)."
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g1_a(_: *const i32) {} // expect: "FvPE": 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g1_b(_: *mut i32) {} // expect: "FvPE": 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g1_c(_: &i32) {} // expect: "FvPE": 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g1_d(_: &mut i32) {} // expect: "FvPE": 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g1_e(_: Box<i32>) {} // expect: "FvPE" 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g1_f(_: NonNull<i32>) {} // expect: "FvPE": 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g1_g(_: *const SomeStruct) {} // expect: "FvPE" 10942 (0x2abe)

struct MyAlloc;
unsafe impl Allocator for MyAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate(layout)
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { Global.deallocate(ptr, layout) }
    }
}
// negative test for "specifically, only Box<T, Global>"
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g1_box_alloc_neg(_: Box<i32, MyAlloc>) {} // expect: "Fv3BoxE": 3916 (0xf4c)

// "usize is ABI-compatible with the uN integer type of the same size, and
// likewise isize is ABI-compatible with the iN integer type of the same size.
// and
// "char is ABI-compatible with u32."
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g2_a(_: i32) {} // expect: "FviE": 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g2_b(_: u64) {} // expect: "FviE": 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g2_c(_: usize) {} // expect: "FviE": 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g2_d(_: isize) {} // expect: "FviE": 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g2_e(_: char) {} // expect: "FviE": 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g2_f(_: bool) {} // expect: "FviE": 2712 (0xa98)

// "Any two types with size 0 and alignment 1 are ABI-compatible."
struct Unit;
enum OneVariant {
    A,
}
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g3_a(_: ()) {} // expect: "FvvE": 61000 (0xee48)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g3_b(_: Unit) {} // expect: "FvvE": 61000 (0xee48)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g3_c(_: PhantomData<i32>) {} // expect: "FvvE": 61000 (0xee48)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g3_d(_: [u8; 0]) {} // expect: "FvvE": 61000 (0xee48)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g3_e(_: OneVariant) {} // expect: "FvvE": 61000 (0xee48)
// 1-ZST, not 4-ZST, must not match
#[repr(align(4))]
struct AlignedMarker;
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g3_neg(_: AlignedMarker) {} // expect: "Fv13AlignedMarkerE": 49590 (0xc1b6)

// "A repr(transparent) type T is ABI-compatible with its unique non-trivial
// field, i.e., the unique field that doesn’t have size 0 and alignment 1 (if
// there is such a field)."
#[repr(transparent)]
struct Wrapper(i32);
#[repr(transparent)]
struct WrapperWithZst(i32, PhantomData<u8>);
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g4_a(_: i32) {} // expect: "FviE": 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g4_b(_: Wrapper) {} // expect: "FviE": 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g4_c(_: WrapperWithZst) {} // expect: "FviE": 2712 (0xa98)
// i32 wrapped in a non-transparent struct
struct NotTransparent(i32);
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g4_neg(_: NotTransparent) {} // expect: "Fv14NotTransparentE": 37756 (0x937c)

// "i32 is ABI-compatible with NonZero<i32>, and similar for all other integer
// types."
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g5_a(_: i32) {} // expect: "FviE": 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g5_b(_: NonZero<i32>) {} // expect: "FviE": 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g5_c(_: i16) {} // expect: "FviE": 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g5_d(_: NonZero<i16>) {} // expect: "FviE": 2712 (0xa98)

// If T is guaranteed to be subject to the null pointer optimization, and E is
// an enum satisfying the following requirements, then T and E are
// ABI-compatible. Such an enum E is called “option-like”.
// * The enum E uses the Rust representation, and is not modified by the align
//   or packed representation modifiers.
// * The enum E has exactly two variants.
// * One variant has exactly one field, of type T.
// * All fields of the other variant are zero-sized with 1-byte alignment.
enum MyOption<T> {
    None,
    Some(T),
}
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g6_ref_a(_: &i32) {} // expect: "FvPE": 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g6_ref_b(_: Option<&i32>) {} // expect: "FvPE": 10942 (0x2abe)

#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g6_fn_a(_: fn()) {} // expect: "FvPE": 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g6_fn_b(_: Option<fn()>) {} // expect: "FvPE": 10942 (0x2abe)

#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g6_nn_a(_: NonNull<i32>) {} // expect: "FvPE": 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g6_nn_b(_: Option<NonNull<i32>>) {} // expect: "FvPE": 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g6_nn_c(_: MyOption<NonNull<i32>>) {} // expect: "FvPE": 10942 (0x2abe)

#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g6_box_a(_: Box<i32>) {} // expect: "FvPE": 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g6_box_b(_: Option<Box<i32>>) {} // expect: "FvPE": 10942 (0x2abe)
// matches g2/g4/g5 ("FviE": 2712 (0xa98)), NOT g6_ref/fn/nn/box ("FvPE": 10942
// (0x2abe))
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g6_nz_a(_: NonZero<i32>) {} // expect: "FviE": 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g6_nz_b(_: Option<NonZero<i32>>) {} // expect: "FviE": 2712 (0xa98)

// Option<i32> has no niche to exploit, NPO is not happening.
// Carries a real discriminant, must NOT match plain i32.
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g6_neg(_: Option<i32>) {} // expect: "Fv6OptionE": 20395 (0x4fab)

// Niche field and an extra field. Must NOT fold to NonZero<i32> (the second
// field changes the ABI)
enum WithExtraField {
    A(NonZero<i32>, i32),
    B,
}
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g6_multifield_neg(_: WithExtraField) {} // expect: "Fv14WithExtraFieldE": 34575 (0x870f)

// More than 2 variants sharing one niche field's spare values.
enum ThreeVariants {
    A(bool),
    B,
    C,
}
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g6_arity_neg(_: ThreeVariants) {} // expect: "Fv13ThreeVariantsE": 9871 (0x268f)
                                                //
// "Any two fn (function pointer) types are ABI-compatible with each other if
// they have the same ABI string or the ABI string only differs in a trailing
// -unwind, independent of the rest of their signature."
//
// This is only true for the first part of the rule, if signature differs, the
// discriminator *must* differ (unless it the function pointer is used as an
// argument to a different function, in which case it is encoded as any other
// pointer like type: `P`).
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn g7_a(_: i32) {} // expect: "FviE": 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C-unwind" fn g7_b(_: i32) {} // expect: "FviE": 2712 (0xa98)
