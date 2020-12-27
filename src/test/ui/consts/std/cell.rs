#![feature(const_refs_to_cell)]

use std::cell::*;

// not ok, because this would create a silent constant with interior mutability.
// the rules could be relaxed in the future
static FOO: Wrap<*mut u32> = Wrap(Cell::new(42).as_ptr());
//~^ ERROR encountered dangling pointer
const FOO_CONST: Wrap<*mut u32> = Wrap(Cell::new(42).as_ptr());
//~^ ERROR encountered dangling pointer

static FOO3: Wrap<Cell<u32>> = Wrap(Cell::new(42));
const FOO3_CONST: Wrap<Cell<u32>> = Wrap(Cell::new(42));

// ok
static FOO4: Wrap<*mut u32> = Wrap(FOO3.0.as_ptr());
const FOO4_CONST: Wrap<*mut u32> = Wrap(FOO3_CONST.0.as_ptr());
//~^ ERROR encountered dangling pointer

// not ok, because the `as_ptr` call takes a reference to a type with interior mutability
// which is not allowed in constants
const FOO2: *mut u32 = Cell::new(42).as_ptr();
//~^ ERROR encountered dangling pointer

struct IMSafeTrustMe(UnsafeCell<u32>);
unsafe impl Send for IMSafeTrustMe {}
unsafe impl Sync for IMSafeTrustMe {}

static BAR: IMSafeTrustMe = IMSafeTrustMe(UnsafeCell::new(5));



struct Wrap<T>(T);
unsafe impl<T> Send for Wrap<T> {}
unsafe impl<T> Sync for Wrap<T> {}

static BAR_PTR: Wrap<*mut u32> = Wrap(BAR.0.get());

fn main() {}
