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

fn main() {}
