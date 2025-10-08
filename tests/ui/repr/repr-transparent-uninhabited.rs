#![feature(never_type)]
#![deny(repr_transparent_external_private_fields)]

enum Void {}

pub type Sized = i32;

#[repr(transparent)]
pub struct T1(!);
#[repr(transparent)]
pub struct T2((), Void);
#[repr(transparent)]
pub struct T3(!, ());

#[repr(transparent)]
pub struct T5(Sized, Void);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types

#[repr(transparent)]
pub struct T6(!, Sized);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types

fn main() {}
