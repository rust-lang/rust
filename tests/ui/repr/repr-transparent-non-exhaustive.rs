#![deny(repr_transparent_external_private_fields)]

//@ aux-build: repr-transparent-non-exhaustive.rs
extern crate repr_transparent_non_exhaustive;

use repr_transparent_non_exhaustive::{
    Private,
    NonExhaustive,
    NonExhaustiveEnum,
    NonExhaustiveVariant,
    ExternalIndirection,
};

pub struct InternalPrivate {
    _priv: (),
}

#[non_exhaustive]
pub struct InternalNonExhaustive;

pub struct InternalIndirection<T> {
    x: T,
}

pub type Sized = i32;

#[repr(transparent)]
pub struct T1(Sized, InternalPrivate);
#[repr(transparent)]
pub struct T2(Sized, InternalNonExhaustive);
#[repr(transparent)]
pub struct T3(Sized, InternalIndirection<(InternalPrivate, InternalNonExhaustive)>);
#[repr(transparent)]
pub struct T4(Sized, ExternalIndirection<(InternalPrivate, InternalNonExhaustive)>);

#[repr(transparent)]
pub struct T5(Sized, Private);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T6(Sized, NonExhaustive);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T7(Sized, NonExhaustiveEnum);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T8(Sized, NonExhaustiveVariant);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T9(Sized, InternalIndirection<Private>);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T10(Sized, InternalIndirection<NonExhaustive>);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T11(Sized, InternalIndirection<NonExhaustiveEnum>);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T12(Sized, InternalIndirection<NonExhaustiveVariant>);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T13(Sized, ExternalIndirection<Private>);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T14(Sized, ExternalIndirection<NonExhaustive>);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T15(Sized, ExternalIndirection<NonExhaustiveEnum>);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T16(Sized, ExternalIndirection<NonExhaustiveVariant>);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T17(NonExhaustive, Sized);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T18(NonExhaustive, NonExhaustive);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T19(NonExhaustive, Private);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T19Flipped(Private, NonExhaustive);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T20(NonExhaustive);
// Okay, since it's the only field.

#[repr(transparent)]
pub struct T21(NonExhaustive, InternalNonExhaustive);
// Okay, since there's only 1 foreign non-exhaustive type.

#[repr(transparent)]
pub struct T21Flipped(InternalNonExhaustive, NonExhaustive);
// Okay, since there's only 1 foreign non-exhaustive type.

#[repr(transparent)]
pub struct T22(NonExhaustive, InternalPrivate);
// Okay, since there's only 1 foreign non-exhaustive type.

#[repr(transparent)]
pub struct T22Flipped(InternalPrivate, NonExhaustive);
// Okay, since there's only 1 foreign non-exhaustive type.

fn main() {}
