//@ check-fail

// Test that `#[rustc_anti_fundamental]` on std traits (Deref, Receiver,
// CoerceUnsized, DispatchFromDyn) prevents implementing them for foreign
// fundamental types like `Pin`.

#![feature(arbitrary_self_types, coerce_unsized, dispatch_from_dyn)]

use std::ops::{CoerceUnsized, Deref, DerefMut, DispatchFromDyn, Receiver};
use std::pin::Pin;

struct LocalType;
struct LocalType2;

// ERROR: cannot implement Deref for Pin<LocalType>
impl Deref for Pin<LocalType> {
    type Target = LocalType;
    fn deref(&self) -> &LocalType {
        unimplemented!()
    }
}
//~^^^^^^ ERROR cannot implement `Deref` for the fundamental type

// ERROR: cannot implement DerefMut for Pin<LocalType>
impl DerefMut for Pin<LocalType> {
    fn deref_mut(&mut self) -> &mut LocalType {
        unimplemented!()
    }
}
//~^^^^^ ERROR cannot implement `DerefMut` for the fundamental type

// ERROR: cannot implement Receiver for Pin<LocalType>
impl Receiver for Pin<LocalType> {
    type Target = LocalType;
}
//~^^^ ERROR cannot implement `std::ops::Receiver` for the fundamental type

// ERROR: cannot implement CoerceUnsized for Pin<LocalType>
impl CoerceUnsized<Pin<LocalType2>> for Pin<LocalType> {}
//~^ ERROR cannot implement `CoerceUnsized` for the fundamental type
//~| ERROR the trait bound `LocalType: CoerceUnsized<LocalType2>` is not satisfied

// ERROR: cannot implement DispatchFromDyn for Pin<LocalType>
impl DispatchFromDyn<Pin<LocalType2>> for Pin<LocalType> {}
//~^ ERROR cannot implement `DispatchFromDyn` for the fundamental type

struct LocalBoxType;

// ERROR: cannot implement Deref for Box<LocalBoxType>
impl Deref for Box<LocalBoxType> {
    type Target = LocalBoxType;
    fn deref(&self) -> &LocalBoxType {
        unimplemented!()
    }
}
//~^^^^^^ ERROR cannot implement `Deref` for the fundamental type

// ERROR: cannot implement DerefMut for Box<LocalBoxType>
impl DerefMut for Box<LocalBoxType> {
    fn deref_mut(&mut self) -> &mut LocalBoxType {
        unimplemented!()
    }
}
//~^^^^^ ERROR cannot implement `DerefMut` for the fundamental type

// ERROR: cannot implement Deref for &Pin<LocalType>
impl Deref for &Pin<LocalType> {
    type Target = LocalType;
    fn deref(&self) -> &LocalType {
        unimplemented!()
    }
}
//~^^^^^^ ERROR cannot implement `Deref` for the fundamental type

// ERROR: cannot implement Receiver for Box<LocalBoxType>
impl Receiver for Box<LocalBoxType> {
    type Target = LocalBoxType;
}
//~^^^ ERROR cannot implement `std::ops::Receiver` for the fundamental type

fn main() {}
