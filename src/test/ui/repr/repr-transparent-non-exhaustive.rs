#![deny(repr_transparent_external_private_fields)]

// aux-build: repr-transparent-non-exhaustive.rs
extern crate repr_transparent_non_exhaustive;

use repr_transparent_non_exhaustive::{Private, NonExhaustive, ExternalIndirection};

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
//~^ ERROR zero-sized fields in repr(transparent) cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T6(Sized, NonExhaustive);
//~^ ERROR zero-sized fields in repr(transparent) cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T7(Sized, InternalIndirection<Private>);
//~^ ERROR zero-sized fields in repr(transparent) cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T8(Sized, InternalIndirection<NonExhaustive>);
//~^ ERROR zero-sized fields in repr(transparent) cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T9(Sized, ExternalIndirection<Private>);
//~^ ERROR zero-sized fields in repr(transparent) cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T10(Sized, ExternalIndirection<NonExhaustive>);
//~^ ERROR zero-sized fields in repr(transparent) cannot contain external non-exhaustive types
//~| WARN this was previously accepted by the compiler

fn main() {}
