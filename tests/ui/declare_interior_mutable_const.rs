#![warn(clippy::declare_interior_mutable_const)]

use std::borrow::Cow;
use std::cell::Cell;
use std::fmt::Display;
use std::sync::atomic::AtomicUsize;
use std::sync::Once;

const ATOMIC: AtomicUsize = AtomicUsize::new(5); //~ ERROR interior mutable
const CELL: Cell<usize> = Cell::new(6); //~ ERROR interior mutable
const ATOMIC_TUPLE: ([AtomicUsize; 1], Vec<AtomicUsize>, u8) = ([ATOMIC], Vec::new(), 7);
//~^ ERROR interior mutable

macro_rules! declare_const {
    ($name:ident: $ty:ty = $e:expr) => {
        const $name: $ty = $e;
    };
}
declare_const!(_ONCE: Once = Once::new()); //~ ERROR interior mutable

// const ATOMIC_REF: &AtomicUsize = &AtomicUsize::new(7); // This will simply trigger E0492.

const INTEGER: u8 = 8;
const STRING: String = String::new();
const STR: &str = "012345";
const COW: Cow<str> = Cow::Borrowed("abcdef");
//^ note: a const item of Cow is used in the `postgres` package.

const NO_ANN: &dyn Display = &70;

static STATIC_TUPLE: (AtomicUsize, String) = (ATOMIC, STRING);
//^ there should be no lints on this line

#[allow(clippy::declare_interior_mutable_const)]
const ONCE_INIT: Once = Once::new();

trait Trait<T>: Copy {
    type NonCopyType;

    const ATOMIC: AtomicUsize; //~ ERROR interior mutable
    const INTEGER: u64;
    const STRING: String;
    const SELF: Self; // (no error)
    const INPUT: T;
    //~^ ERROR interior mutable
    //~| HELP consider requiring `T` to be `Copy`
    const ASSOC: Self::NonCopyType;
    //~^ ERROR interior mutable
    //~| HELP consider requiring `<Self as Trait<T>>::NonCopyType` to be `Copy`

    const AN_INPUT: T = Self::INPUT;
    //~^ ERROR interior mutable
    //~| ERROR consider requiring `T` to be `Copy`
    declare_const!(ANOTHER_INPUT: T = Self::INPUT); //~ ERROR interior mutable
}

trait Trait2 {
    type CopyType: Copy;

    const SELF_2: Self;
    //~^ ERROR interior mutable
    //~| HELP consider requiring `Self` to be `Copy`
    const ASSOC_2: Self::CopyType; // (no error)
}

// we don't lint impl of traits, because an impl has no power to change the interface.
impl Trait<u32> for u64 {
    type NonCopyType = u16;

    const ATOMIC: AtomicUsize = AtomicUsize::new(9);
    const INTEGER: u64 = 10;
    const STRING: String = String::new();
    const SELF: Self = 11;
    const INPUT: u32 = 12;
    const ASSOC: Self::NonCopyType = 13;
}

struct Local<T, U>(T, U);

impl<T: Trait2 + Trait<u32>, U: Trait2> Local<T, U> {
    const ASSOC_3: AtomicUsize = AtomicUsize::new(14); //~ ERROR interior mutable
    const COW: Cow<'static, str> = Cow::Borrowed("tuvwxy");
    const T_SELF: T = T::SELF_2;
    const U_SELF: U = U::SELF_2;
    //~^ ERROR interior mutable
    //~| HELP consider requiring `U` to be `Copy`
    const T_ASSOC: T::NonCopyType = T::ASSOC;
    //~^ ERROR interior mutable
    //~| HELP consider requiring `<T as Trait<u32>>::NonCopyType` to be `Copy`
    const U_ASSOC: U::CopyType = U::ASSOC_2;
}

fn main() {}
