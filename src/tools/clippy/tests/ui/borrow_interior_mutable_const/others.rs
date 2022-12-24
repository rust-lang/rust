#![warn(clippy::borrow_interior_mutable_const)]
#![allow(clippy::declare_interior_mutable_const, clippy::needless_borrow)]
#![allow(const_item_mutation)]

use std::borrow::Cow;
use std::cell::{Cell, UnsafeCell};
use std::fmt::Display;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Once;

const ATOMIC: AtomicUsize = AtomicUsize::new(5);
const CELL: Cell<usize> = Cell::new(6);
const ATOMIC_TUPLE: ([AtomicUsize; 1], Vec<AtomicUsize>, u8) = ([ATOMIC], Vec::new(), 7);
const INTEGER: u8 = 8;
const STRING: String = String::new();
const STR: &str = "012345";
const COW: Cow<str> = Cow::Borrowed("abcdef");
const NO_ANN: &dyn Display = &70;
static STATIC_TUPLE: (AtomicUsize, String) = (ATOMIC, STRING);
const ONCE_INIT: Once = Once::new();

// This is just a pointer that can be safely dereferenced,
// it's semantically the same as `&'static T`;
// but it isn't allowed to make a static reference from an arbitrary integer value at the moment.
// For more information, please see the issue #5918.
pub struct StaticRef<T> {
    ptr: *const T,
}

impl<T> StaticRef<T> {
    /// Create a new `StaticRef` from a raw pointer
    ///
    /// ## Safety
    ///
    /// Callers must pass in a reference to statically allocated memory which
    /// does not overlap with other values.
    pub const unsafe fn new(ptr: *const T) -> StaticRef<T> {
        StaticRef { ptr }
    }
}

impl<T> std::ops::Deref for StaticRef<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.ptr }
    }
}

// use a tuple to make sure referencing a field behind a pointer isn't linted.
const CELL_REF: StaticRef<(UnsafeCell<u32>,)> = unsafe { StaticRef::new(std::ptr::null()) };

fn main() {
    ATOMIC.store(1, Ordering::SeqCst); //~ ERROR interior mutability
    assert_eq!(ATOMIC.load(Ordering::SeqCst), 5); //~ ERROR interior mutability

    let _once = ONCE_INIT;
    let _once_ref = &ONCE_INIT; //~ ERROR interior mutability
    let _once_ref_2 = &&ONCE_INIT; //~ ERROR interior mutability
    let _once_ref_4 = &&&&ONCE_INIT; //~ ERROR interior mutability
    let _once_mut = &mut ONCE_INIT; //~ ERROR interior mutability
    let _atomic_into_inner = ATOMIC.into_inner();
    // these should be all fine.
    let _twice = (ONCE_INIT, ONCE_INIT);
    let _ref_twice = &(ONCE_INIT, ONCE_INIT);
    let _ref_once = &(ONCE_INIT, ONCE_INIT).0;
    let _array_twice = [ONCE_INIT, ONCE_INIT];
    let _ref_array_twice = &[ONCE_INIT, ONCE_INIT];
    let _ref_array_once = &[ONCE_INIT, ONCE_INIT][0];

    // referencing projection is still bad.
    let _ = &ATOMIC_TUPLE; //~ ERROR interior mutability
    let _ = &ATOMIC_TUPLE.0; //~ ERROR interior mutability
    let _ = &(&&&&ATOMIC_TUPLE).0; //~ ERROR interior mutability
    let _ = &ATOMIC_TUPLE.0[0]; //~ ERROR interior mutability
    let _ = ATOMIC_TUPLE.0[0].load(Ordering::SeqCst); //~ ERROR interior mutability
    let _ = &*ATOMIC_TUPLE.1;
    let _ = &ATOMIC_TUPLE.2;
    let _ = (&&&&ATOMIC_TUPLE).0;
    let _ = (&&&&ATOMIC_TUPLE).2;
    let _ = ATOMIC_TUPLE.0;
    let _ = ATOMIC_TUPLE.0[0]; //~ ERROR interior mutability
    let _ = ATOMIC_TUPLE.1.into_iter();
    let _ = ATOMIC_TUPLE.2;
    let _ = &{ ATOMIC_TUPLE };

    CELL.set(2); //~ ERROR interior mutability
    assert_eq!(CELL.get(), 6); //~ ERROR interior mutability

    assert_eq!(INTEGER, 8);
    assert!(STRING.is_empty());

    let a = ATOMIC;
    a.store(4, Ordering::SeqCst);
    assert_eq!(a.load(Ordering::SeqCst), 4);

    STATIC_TUPLE.0.store(3, Ordering::SeqCst);
    assert_eq!(STATIC_TUPLE.0.load(Ordering::SeqCst), 3);
    assert!(STATIC_TUPLE.1.is_empty());

    assert_eq!(NO_ANN.to_string(), "70"); // should never lint this.

    let _ = &CELL_REF.0;
}
