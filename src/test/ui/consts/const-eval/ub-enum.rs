#![feature(const_transmute, never_type)]
#![allow(const_err)] // make sure we cannot allow away the errors tested here

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
//~^ ERROR is undefined behavior

const BAD_ENUM_WRAPPED: Wrap<Enum> = unsafe { mem::transmute(&1) };
//~^ ERROR is undefined behavior

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
//~^ ERROR is undefined behavior
// something wrapping the enum so that we test layout first, not enum
const BAD_ENUM2_WRAPPED: Wrap<Enum2> = unsafe { mem::transmute(&0) };
//~^ ERROR is undefined behavior

// Undef enum discriminant.
#[repr(C)]
union MaybeUninit<T: Copy> {
    uninit: (),
    init: T,
}
const BAD_ENUM2_UNDEF : Enum2 = unsafe { MaybeUninit { uninit: () }.init };
//~^ ERROR is undefined behavior

// Pointer value in an enum with a niche that is not just 0.
const BAD_ENUM2_OPTION_PTR: Option<Enum2> = unsafe { mem::transmute(&0) };
//~^ ERROR is undefined behavior

// # valid discriminant for uninhabited variant

// An enum with 3 variants of which some are uninhabited -- so the uninhabited variants *do*
// have a discriminant.
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
const BAD_UNINHABITED_WITH_DATA1: Result<(i32, Never), (i32, !)> = unsafe { mem::transmute(1u64) };
//~^ ERROR is undefined behavior
const BAD_UNINHABITED_WITH_DATA2: Result<(i32, !), (i32, Never)> = unsafe { mem::transmute(1u64) };
//~^ ERROR is undefined behavior

fn main() {
}
