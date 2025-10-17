#![deny(repr_transparent_non_zst_fields)]

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
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain `repr(C)` types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T6(ReprC1Zst, Sized);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain `repr(C)` types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T7(T1, Sized); // still wrong, even when the repr(C) is hidden inside another type
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain `repr(C)` types
//~| WARN this was previously accepted by the compiler

fn main() {}
