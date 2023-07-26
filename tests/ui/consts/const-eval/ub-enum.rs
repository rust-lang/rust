// stderr-per-bitwidth
// Strip out raw byte dumps to make comparison platform-independent:
// normalize-stderr-test "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
// normalize-stderr-test "([0-9a-f][0-9a-f] |╾─*a(lloc)?[0-9]+(\+[a-z0-9]+)?─*╼ )+ *│.*" -> "HEX_DUMP"
#![feature(never_type, const_discriminant)]
#![allow(invalid_value)]

use std::mem;

#[repr(transparent)]
#[derive(Copy, Clone)]
struct Wrap<T>(T);

#[derive(Copy, Clone)]
enum Never {}

// # simple enum with discriminant 0

#[repr(usize)]
#[derive(Copy, Clone)]
enum Enum {
    A = 0,
}

const GOOD_ENUM: Enum = unsafe { mem::transmute(0usize) };

const BAD_ENUM: Enum = unsafe { mem::transmute(1usize) };
//~^ ERROR is undefined behavior

const BAD_ENUM_PTR: Enum = unsafe { mem::transmute(&1) };
//~^ ERROR evaluation of constant value failed

const BAD_ENUM_WRAPPED: Wrap<Enum> = unsafe { mem::transmute(&1) };
//~^ ERROR evaluation of constant value failed

// # simple enum with discriminant 2

// (Potentially) invalid enum discriminant
#[repr(usize)]
#[derive(Copy, Clone)]
enum Enum2 {
    A = 2,
}

const BAD_ENUM2: Enum2 = unsafe { mem::transmute(0usize) };
//~^ ERROR is undefined behavior
const BAD_ENUM2_PTR: Enum2 = unsafe { mem::transmute(&0) };
//~^ ERROR evaluation of constant value failed
// something wrapping the enum so that we test layout first, not enum
const BAD_ENUM2_WRAPPED: Wrap<Enum2> = unsafe { mem::transmute(&0) };
//~^ ERROR evaluation of constant value failed

// Undef enum discriminant.
#[repr(C)]
union MaybeUninit<T: Copy> {
    uninit: (),
    init: T,
}
const BAD_ENUM2_UNDEF : Enum2 = unsafe { MaybeUninit { uninit: () }.init };
//~^ ERROR evaluation of constant value failed
//~| uninitialized

// Pointer value in an enum with a niche that is not just 0.
const BAD_ENUM2_OPTION_PTR: Option<Enum2> = unsafe { mem::transmute(&0) };
//~^ ERROR evaluation of constant value failed

// # valid discriminant for uninhabited variant

// An enum with uninhabited variants but also at least 2 inhabited variants -- so the uninhabited
// variants *do* have a discriminant.
enum UninhDiscriminant {
    A,
    B(!),
    C,
    D(Never),
}

const GOOD_INHABITED_VARIANT1: UninhDiscriminant = unsafe { mem::transmute(0u8) }; // variant A
const GOOD_INHABITED_VARIANT2: UninhDiscriminant = unsafe { mem::transmute(2u8) }; // variant C

const BAD_UNINHABITED_VARIANT1: UninhDiscriminant = unsafe { mem::transmute(1u8) };
//~^ ERROR is undefined behavior
const BAD_UNINHABITED_VARIANT2: UninhDiscriminant = unsafe { mem::transmute(3u8) };
//~^ ERROR is undefined behavior

// # other

// Invalid enum field content (mostly to test printing of paths for enum tuple
// variants and tuples).
// Need to create something which does not clash with enum layout optimizations.
const BAD_OPTION_CHAR: Option<(char, char)> = Some(('x', unsafe { mem::transmute(!0u32) }));
//~^ ERROR is undefined behavior

// All variants are uninhabited but also have data.
// Use `0` as constant to make behavior endianness-independent.
const BAD_UNINHABITED_WITH_DATA1: Result<(i32, Never), (i32, !)> = unsafe { mem::transmute(0u64) };
//~^ ERROR evaluation of constant value failed
const BAD_UNINHABITED_WITH_DATA2: Result<(i32, !), (i32, Never)> = unsafe { mem::transmute(0u64) };
//~^ ERROR evaluation of constant value failed

const TEST_ICE_89765: () = {
    // This is a regression test for https://github.com/rust-lang/rust/issues/89765.
    unsafe { std::mem::discriminant(&*(&() as *const () as *const Never)); };
    //~^ inside `TEST_ICE_89765`
};

fn main() {
}
