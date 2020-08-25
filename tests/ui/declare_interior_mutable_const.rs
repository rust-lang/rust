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

struct Wrapper<T>(T);

trait Trait<T: Trait2<AssocType5 = AtomicUsize>> {
    type AssocType;
    type AssocType2;
    type AssocType3;

    const ATOMIC: AtomicUsize; //~ ERROR interior mutable
    const INTEGER: u64;
    const STRING: String;
    const SELF: Self;
    const INPUT: T;
    const INPUT_ASSOC: T::AssocType4;
    const INPUT_ASSOC_2: T::AssocType5; //~ ERROR interior mutable
    const ASSOC: Self::AssocType;
    const ASSOC_2: Self::AssocType2;
    const WRAPPED_ASSOC_2: Wrapper<Self::AssocType2>;
    const WRAPPED_ASSOC_3: Wrapper<Self::AssocType3>;

    const AN_INPUT: T = Self::INPUT;
    declare_const!(ANOTHER_INPUT: T = Self::INPUT);
    declare_const!(ANOTHER_ATOMIC: AtomicUsize = Self::ATOMIC); //~ ERROR interior mutable
}

trait Trait2 {
    type AssocType4;
    type AssocType5;

    const SELF_2: Self;
    const ASSOC_4: Self::AssocType4;
}

impl<T: Trait2<AssocType5 = AtomicUsize>> Trait<T> for u64 {
    type AssocType = u16;
    type AssocType2 = AtomicUsize;
    type AssocType3 = T;

    const ATOMIC: AtomicUsize = AtomicUsize::new(9);
    const INTEGER: u64 = 10;
    const STRING: String = String::new();
    const SELF: Self = 11;
    const INPUT: T = T::SELF_2;
    const INPUT_ASSOC: T::AssocType4 = T::ASSOC_4;
    const INPUT_ASSOC_2: T::AssocType5 = AtomicUsize::new(16);
    const ASSOC: Self::AssocType = 13;
    const ASSOC_2: Self::AssocType2 = AtomicUsize::new(15); //~ ERROR interior mutable
    const WRAPPED_ASSOC_2: Wrapper<Self::AssocType2> = Wrapper(AtomicUsize::new(16)); //~ ERROR interior mutable
    const WRAPPED_ASSOC_3: Wrapper<Self::AssocType3> = Wrapper(T::SELF_2);
}

struct Local<T, U>(T, U);

impl<T: Trait<U>, U: Trait2<AssocType5 = AtomicUsize>> Local<T, U> {
    const ASSOC_5: AtomicUsize = AtomicUsize::new(14); //~ ERROR interior mutable
    const COW: Cow<'static, str> = Cow::Borrowed("tuvwxy");
    const U_SELF: U = U::SELF_2;
    const T_ASSOC: T::AssocType = T::ASSOC;
    const U_ASSOC: U::AssocType5 = AtomicUsize::new(17); //~ ERROR interior mutable
}

fn main() {}
