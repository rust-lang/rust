// Test how function pointer type discriminator calculation interacts with function pointer ABI
// compatibility rules (https://doc.rust-lang.org/std/primitive.fn.html#abi-compatibility).
//
// Function pointers are signed with the discriminator of the function they point to, and a call
// authenticates with the discriminator of the type it is made through.
//
// The test is grouped into 3 sections, based on the type compatibility (marked by comments in
// code):
//
// 1. Types that are ABI-compatible , so the discriminators MUST match. A failure here is a
//    soundness bug.
// 2. Not ABI-compatible types, but with equal discriminators, because the Clang encoding is coarser
//    than Rust's rules, for example: every integer type encodes as `i`. These MUST match to match
//    Clang. A failure here breaks interoperability with C.
// 3. Types that are not ABI-compatible, and distinct. Calls through these signatures are UB;
//    distinct discriminators are what makes such a call trap.
//
// Each function's encoding and hash are checked by its annotations
// (#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]).
//
// In sections 1 and 2, every function in a group has the same encoding; in section 3, each
// function differs from the group it resembles.
//
// On the rule that "any two fn (function pointer) types are ABI-compatible with each other": it is
// about function pointers passed as values, and is honoured, as every function pointer argument
// encodes as `P` (group 1.7). It does not make a call through a different signature valid: calling
// a `fn(i32)` through `fn()` is still UB, so the two signatures get distinct discriminators
// (group 3.1).

#![crate_type = "lib"]
#![feature(rustc_attrs)]
#![feature(allocator_ext)]
#![allow(dead_code)]
#![allow(improper_ctypes_definitions)]

use std::alloc::{AllocError, Allocator, Global, Layout};
use std::marker::PhantomData;
use std::num::NonZero;
use std::ptr::NonNull;

struct SomeStruct {
    _x: i32,
}

struct Unit;

enum OneVariant {
    A,
}

#[repr(transparent)]
struct TrivialWrapper(());

#[repr(transparent)]
struct Wrapper(i32);

#[repr(transparent)]
struct WrapperWithZst(i32, PhantomData<u8>);

enum MyOption<T> {
    None,
    Some(T),
}

mod m1 {
    pub struct S(pub i32);
}

mod m2 {
    pub struct S(pub u8);
}

struct MyAlloc;

unsafe impl Allocator for MyAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate(layout)
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { Global.deallocate(ptr, layout) }
    }
}

#[repr(align(4))]
struct AlignedMarker;

struct NotTransparent(i32);

enum WithExtraField {
    A(NonZero<i32>, i32),
    B,
}

enum ThreeVariants {
    A(bool),
    B,
    C,
}

// 1. Types that are ABI-compatible , so the discriminators MUST match. A failure here is a
//    soundness bug.

// 1.1 "*const T, *mut T, &T, &mut T, Box<T> (specifically, only Box<T, Global>), and NonNull<T>
// are all ABI-compatible with each other for all T. They are also ABI-compatible with each other
// for different T if they have the same metadata type." Here thin pointers (metadata `()`).
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn p1_a(_: *const i32) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn p1_b(_: *mut i32) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn p1_c(_: &i32) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn p1_d(_: &mut i32) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn p1_e(_: Box<i32>) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn p1_f(_: NonNull<i32>) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn p1_g(_: *const SomeStruct) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)

// 1.2 The same, for pointers whose metadata is `usize` (slices and `str`).
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn p2_a(_: *const [i32]) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn p2_b(_: &[u8]) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn p2_c(_: *const str) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn p2_d(_: Box<[u16]>) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)

// 1.3 "usize is ABI-compatible with the uN integer type of the same size, and likewise isize is
// ABI-compatible with the iN integer type of the same size." and "char is ABI-compatible with
// u32."
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn i1_a(_: usize) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn i1_b(_: u64) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn i1_c(_: isize) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn i1_d(_: i64) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn i1_e(_: char) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn i1_f(_: u32) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)

// 1.4 "Any two types with size 0 and alignment 1 are ABI-compatible.", including tuples and
// arrays of them, `repr(transparent)` wrappers of them, and `!`.
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn z1_a(_: ()) {}
//~^ ERROR encoding: "FvvE"
//~| ERROR hash: 61000 (0xee48)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn z1_b(_: Unit) {}
//~^ ERROR encoding: "FvvE"
//~| ERROR hash: 61000 (0xee48)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn z1_c(_: PhantomData<i32>) {}
//~^ ERROR encoding: "FvvE"
//~| ERROR hash: 61000 (0xee48)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn z1_d(_: [u8; 0]) {}
//~^ ERROR encoding: "FvvE"
//~| ERROR hash: 61000 (0xee48)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn z1_e(_: OneVariant) {}
//~^ ERROR encoding: "FvvE"
//~| ERROR hash: 61000 (0xee48)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn z1_f(_: ((), PhantomData<u8>)) {}
//~^ ERROR encoding: "FvvE"
//~| ERROR hash: 61000 (0xee48)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn z1_g(_: [(); 3]) {}
//~^ ERROR encoding: "FvvE"
//~| ERROR hash: 61000 (0xee48)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn z1_h(_: TrivialWrapper) {}
//~^ ERROR encoding: "FvvE"
//~| ERROR hash: 61000 (0xee48)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn z1_i(_: !) {}
//~^ ERROR encoding: "FvvE"
//~| ERROR hash: 61000 (0xee48)

// 1.5 "A repr(transparent) type T is ABI-compatible with its unique non-trivial field", and
// "i32 is ABI-compatible with NonZero<i32>, and similar for all other integer types."
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn w1_a(_: i32) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn w1_b(_: Wrapper) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn w1_c(_: WrapperWithZst) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn w1_d(_: NonZero<i32>) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)

// 1.6 "If T is guaranteed to be subject to the null pointer optimization, and E is an enum
// satisfying the following requirements, then T and E are ABI-compatible." (option-like enums)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn o1_a(_: Option<&i32>) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn o1_b(_: Option<NonNull<i32>>) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn o1_c(_: MyOption<NonNull<i32>>) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn o1_d(_: Option<Box<i32>>) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn o1_e(_: Option<fn()>) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)

// 1.6 (continued) The same for an integer `T`.
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn o2_a(_: NonZero<i32>) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn o2_b(_: Option<NonZero<i32>>) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)

// 1.7 "Any two fn (function pointer) types are ABI-compatible with each other if they have the
// same ABI string or the ABI string only differs in a trailing -unwind, independent of the rest of
// their signature." This is about passing function pointers as values.
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn f1_a(_: fn()) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn f1_b(_: fn(i32) -> i32) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn f1_c(_: extern "C" fn()) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn f1_d(_: extern "C-unwind" fn(i32)) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)

// 1.8 Return position: trivial-ABI return types, including `!`, are ABI-compatible with `()`.
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn r1_a() {}
//~^ ERROR encoding: "FvE"
//~| ERROR hash: 18983 (0x4a27)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn r1_b() -> () {}
//~^ ERROR encoding: "FvE"
//~| ERROR hash: 18983 (0x4a27)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn r1_c() -> ! { loop {} }
//~^ ERROR encoding: "FvE"
//~| ERROR hash: 18983 (0x4a27)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn r1_d() -> Unit { Unit }
//~^ ERROR encoding: "FvE"
//~| ERROR hash: 18983 (0x4a27)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn r1_e() -> PhantomData<u8> { PhantomData }
//~^ ERROR encoding: "FvE"
//~| ERROR hash: 18983 (0x4a27)

// 1.8 (continued) Pointer-like return types.
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn r2_a() -> *const i32 { std::ptr::null() }
//~^ ERROR encoding: "FPE"
//~| ERROR hash: 34128 (0x8550)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn r2_b() -> Option<&'static i32> { None }
//~^ ERROR encoding: "FPE"
//~| ERROR hash: 34128 (0x8550)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn r2_c() -> Option<NonNull<i32>> { None }
//~^ ERROR encoding: "FPE"
//~| ERROR hash: 34128 (0x8550)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn r2_d() -> Option<Box<i32>> { None }
//~^ ERROR encoding: "FPE"
//~| ERROR hash: 34128 (0x8550)

// 1.8 (continued) Integer-like return types.
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn r3_a() -> u32 { 0 }
//~^ ERROR encoding: "FiE"
//~| ERROR hash: 42271 (0xa51f)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn r3_b() -> char { 'a' }
//~^ ERROR encoding: "FiE"
//~| ERROR hash: 42271 (0xa51f)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn r3_c() -> NonZero<u32> { NonZero::new(1).unwrap() }
//~^ ERROR encoding: "FiE"
//~| ERROR hash: 42271 (0xa51f)

// 1.9 ABI strings: a trailing `-unwind` does not matter. `"system"` is `"C"` on this target.
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn a1_a(_: i32) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C-unwind" fn a1_b(_: i32) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "system" fn a1_c(_: i32) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "system-unwind" fn a1_d(_: i32) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)

// 1.10 `unsafe` is not part of the ABI.
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn s1_a(_: i32) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
unsafe extern "C" fn s1_b(_: i32) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)

// 2. Not ABI-compatible types, but with equal discriminators, because the Clang encoding is coarser
//    than Rust's rules, for example: every integer type encodes as `i`. These MUST match to match
//    Clang. A failure here breaks interoperability with C.

// 2.1 Clang encodes every integer type, `bool` and `char` as `i`, whatever its size.
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn c1_a(_: i32) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn c1_b(_: bool) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn c1_c(_: u8) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn c1_d(_: i16) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn c1_e(_: NonZero<i16>) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)

// 2.2 Every pointer-like type encodes as `P`, whatever its metadata: C has no wide pointers.
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn c2_a(_: *const i32) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn c2_b(_: *const [i32]) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)

// 2.3 Arrays are encoded without their length.
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn c3_a(_: [i32; 4]) {}
//~^ ERROR encoding: "FvAiE"
//~| ERROR hash: 35183 (0x896f)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn c3_b(_: [i32; 8]) {}
//~^ ERROR encoding: "FvAiE"
//~| ERROR hash: 35183 (0x896f)

// 2.4 Structs are encoded by name only, so two different structs with the same name collide.
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn c4_a(_: m1::S) {}
//~^ ERROR encoding: "Fv1SE"
//~| ERROR hash: 61901 (0xf1cd)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn c4_b(_: m2::S) {}
//~^ ERROR encoding: "Fv1SE"
//~| ERROR hash: 61901 (0xf1cd)

// 3. Types that are not ABI-compatible, and distinct. Calls through these signatures are UB;
//    distinct discriminators are what makes such a call trap.

// 3.1 Different signatures. In particular, a `fn(i32)` called through `fn()` traps.
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn d1_a() {}
//~^ ERROR encoding: "FvE"
//~| ERROR hash: 18983 (0x4a27)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn d1_b(_: i32) {}
//~^ ERROR encoding: "FviE"
//~| ERROR hash: 2712 (0xa98)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn d1_c() -> i32 { 0 }
//~^ ERROR encoding: "FiE"
//~| ERROR hash: 42271 (0xa51f)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn d1_d(_: *const i32) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn d1_e(_: f32) {}
//~^ ERROR encoding: "FvfE"
//~| ERROR hash: 21613 (0x546d)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn d1_f(_: f64) {}
//~^ ERROR encoding: "FvdE"
//~| ERROR hash: 7718 (0x1e26)

// 3.2 Near misses of the rules in section 1. Each is distinct from the group it resembles:
// * `Box` with an allocator other than `Global` (1.1);
// * a ZST with alignment 4 (1.4);
// * a non-transparent wrapper (1.5);
// * `Option<i32>`, which has no niche (1.6);
// * `Option<*mut T>`, as a raw pointer has no niche (1.6)
// * an enum whose niche variant has a second field (1.6);
// * an enum with more than two variants (1.6).
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn n1_a(_: Box<i32, MyAlloc>) {}
//~^ ERROR encoding: "Fv3BoxE"
//~| ERROR hash: 3916 (0xf4c)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn n1_b(_: AlignedMarker) {}
//~^ ERROR encoding: "Fv13AlignedMarkerE"
//~| ERROR hash: 49590 (0xc1b6)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn n1_c(_: NotTransparent) {}
//~^ ERROR encoding: "Fv14NotTransparentE"
//~| ERROR hash: 37756 (0x937c)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn n1_d(_: Option<i32>) {}
//~^ ERROR encoding: "Fv6OptionE"
//~| ERROR hash: 20395 (0x4fab)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn n1_e(_: Option<*mut i32>) {}
//~^ ERROR encoding: "Fv6OptionE"
//~| ERROR hash: 20395 (0x4fab)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn n1_f(_: WithExtraField) {}
//~^ ERROR encoding: "Fv14WithExtraFieldE"
//~| ERROR hash: 34575 (0x870f)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn n1_g(_: ThreeVariants) {}
//~^ ERROR encoding: "Fv13ThreeVariantsE"
//~| ERROR hash: 9871 (0x268f)

// 3.3 Only an `Option` with the layout of its payload is peeled. `Option<fn()>` has the layout of
// `fn()` (group 1.6), but it uses up the niche of the function pointer, so `Option<Option<fn()>>`
// needs a tag. It is not ABI-compatible with `fn()`.
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn q1_a(_: Option<extern "C" fn()>) {}
//~^ ERROR encoding: "FvPE"
//~| ERROR hash: 10942 (0x2abe)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn q1_b(_: Option<Option<extern "C" fn()>>) {}
//~^ ERROR encoding: "Fv6OptionE"
//~| ERROR hash: 20395 (0x4fab)
