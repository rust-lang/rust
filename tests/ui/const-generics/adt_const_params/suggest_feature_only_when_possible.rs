// Test that when adt_const_params is not enabled, we suggest adding the feature only when
// it would be possible for the type to be used as a const generic or when it's likely
// possible for the user to fix their type to be used.

// Can never be used as const generics.
fn uwu_0<const N: &'static mut ()>() {}
//~^ ERROR: forbidden as the type of a const generic
//~| HELP: add `#![feature(adt_const_params)]`
//~| HELP: add `#![feature(adt_const_params)]`
//~| HELP: add `#![feature(adt_const_params)]`
//~| HELP: add `#![feature(unsized_const_params)]`
//~| HELP: add `#![feature(unsized_const_params)]`

// Needs the feature but can be used, so suggest adding the feature.
fn owo_0<const N: &'static u32>() {}
//~^ ERROR: forbidden as the type of a const generic

// Can only be used in const generics with changes.
struct Meow {
    meow: u8,
}

fn meow_0<const N: Meow>() {}
//~^ ERROR: forbidden as the type of a const generic
fn meow_1<const N: &'static Meow>() {}
//~^ ERROR: forbidden as the type of a const generic
fn meow_2<const N: [Meow; 100]>() {}
//~^ ERROR: forbidden as the type of a const generic
fn meow_3<const N: (Meow, u8)>() {}
//~^ ERROR: forbidden as the type of a const generic

// This is suboptimal that it thinks it can be used
// but better to suggest the feature to the user.
fn meow_4<const N: (Meow, String)>() {}
//~^ ERROR: forbidden as the type of a const generic

// Non-local ADT that does not impl `ConstParamTy`
fn nya_0<const N: String>() {}
//~^ ERROR: forbidden as the type of a const generic
fn nya_1<const N: Vec<u32>>() {}
//~^ ERROR: forbidden as the type of a const generic

fn main() {}
