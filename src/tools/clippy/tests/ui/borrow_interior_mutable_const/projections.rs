#![deny(clippy::borrow_interior_mutable_const)]
#![deny(clippy::declare_interior_mutable_const)]

// Inspired by https://github.com/rust-lang/rust/pull/130543#issuecomment-2364828139

use std::cell::UnsafeCell;

trait Trait {
    type Assoc;
}

type Assoc<T> = <T as Trait>::Assoc;

impl Trait for u8 {
    type Assoc = UnsafeCell<u8>;
}

impl Trait for () {
    type Assoc = ();
}

enum MaybeMutable {
    Mutable(Assoc<u8>),
    Immutable(Assoc<()>),
}

const CELL: Assoc<u8> = UnsafeCell::new(0); //~ ERROR: interior mutable
const UNIT: Assoc<()> = ();
const MUTABLE: MaybeMutable = MaybeMutable::Mutable(CELL); //~ ERROR: interior mutable
const IMMUTABLE: MaybeMutable = MaybeMutable::Immutable(UNIT);

fn print_ref<T>(t: &T) {
    let p: *const T = t;
    println!("{p:p}")
}

fn main() {
    print_ref(&CELL); //~ ERROR: interior mutability
    print_ref(&UNIT);
    print_ref(&MUTABLE); //~ ERROR: interior mutability
    print_ref(&IMMUTABLE);
}
