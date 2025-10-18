#![deny(repr_transparent_external_private_fields)]

#[repr(C)]
pub struct ReprC1Zst {
    pub _f: (),
}

pub type Sized = i32;

#[repr(transparent)]
pub struct T1(ReprC1Zst);
#[repr(transparent)]
pub struct T2((), ReprC1Zst);
#[repr(transparent)]
pub struct T3(ReprC1Zst, ());

#[repr(transparent)]
pub struct T5(Sized, ReprC1Zst);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types

#[repr(transparent)]
pub struct T6(ReprC1Zst, Sized);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types

#[repr(transparent)]
pub struct T7(T1, Sized); // still wrong, even when the repr(C) is hidden inside another type
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types

fn main() {}
