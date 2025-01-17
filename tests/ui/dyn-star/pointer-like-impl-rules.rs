//@ check-fail

#![feature(extern_types)]
#![feature(pointer_like_trait)]

use std::marker::PointerLike;

struct NotReprTransparent;
impl PointerLike for NotReprTransparent {}
//~^ ERROR: implementation must be applied to type that
//~| NOTE: the struct `NotReprTransparent` is not `repr(transparent)`

#[repr(transparent)]
struct FieldIsPl(usize);
impl PointerLike for FieldIsPl {}

#[repr(transparent)]
struct FieldIsPlAndHasOtherField(usize, ());
impl PointerLike for FieldIsPlAndHasOtherField {}

#[repr(transparent)]
struct FieldIsNotPl(u8);
impl PointerLike for FieldIsNotPl {}
//~^ ERROR: implementation must be applied to type that
//~| NOTE: the field `0` of struct `FieldIsNotPl` does not implement `PointerLike`

#[repr(transparent)]
struct GenericFieldIsNotPl<T>(T);
impl<T> PointerLike for GenericFieldIsNotPl<T> {}
//~^ ERROR: implementation must be applied to type that
//~| NOTE: the field `0` of struct `GenericFieldIsNotPl<T>` does not implement `PointerLike`

#[repr(transparent)]
struct GenericFieldIsPl<T>(T);
impl<T: PointerLike> PointerLike for GenericFieldIsPl<T> {}

#[repr(transparent)]
struct IsZeroSized(());
impl PointerLike for IsZeroSized {}
//~^ ERROR: implementation must be applied to type that
//~| NOTE: the struct `IsZeroSized` is `repr(transparent)`, but does not have a non-trivial field

trait SomeTrait {}
impl PointerLike for dyn SomeTrait {}
//~^ ERROR: implementation must be applied to type that
//~| NOTE: types of dynamic or unknown size

extern "C" {
    type ExternType;
}
impl PointerLike for ExternType {}
//~^ ERROR: implementation must be applied to type that
//~| NOTE: types of dynamic or unknown size

struct LocalSizedType(&'static str);
struct LocalUnsizedType(str);

// This is not a special error but a normal coherence error,
// which should still happen.
impl PointerLike for &LocalSizedType {}
//~^ ERROR: conflicting implementations of trait `PointerLike`
//~| NOTE: conflicting implementation in crate `core`

impl PointerLike for &LocalUnsizedType {}
//~^ ERROR: implementation must be applied to type that
//~| NOTE: references to dynamically-sized types are too large to be `PointerLike`

impl PointerLike for Box<LocalSizedType> {}
//~^ ERROR: conflicting implementations of trait `PointerLike`
//~| NOTE: conflicting implementation in crate `alloc`

impl PointerLike for Box<LocalUnsizedType> {}
//~^ ERROR: implementation must be applied to type that
//~| NOTE: boxes of dynamically-sized types are too large to be `PointerLike`

fn expects_pointer_like(x: impl PointerLike) {}

fn main() {
    expects_pointer_like(FieldIsPl(1usize));
    expects_pointer_like(FieldIsPlAndHasOtherField(1usize, ()));
    expects_pointer_like(GenericFieldIsPl(1usize));
}
