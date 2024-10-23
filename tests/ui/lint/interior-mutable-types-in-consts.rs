//@ check-pass

#![feature(reentrant_lock)]

use std::sync::{Once, Barrier, Condvar, LazyLock, Mutex, OnceLock, ReentrantLock, RwLock};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicI32, AtomicU32};

const A: Once = Once::new();
//~^ WARN interior mutability in `const` items
const B: Barrier = Barrier::new(0);
//~^ WARN interior mutability in `const` items
const C: Condvar = Condvar::new();
//~^ WARN interior mutability in `const` items
const D: LazyLock<i32> = LazyLock::new(|| 0);
//~^ WARN interior mutability in `const` items
const E: Mutex<i32> = Mutex::new(0);
//~^ WARN interior mutability in `const` items
const F: OnceLock<i32> = OnceLock::new();
//~^ WARN interior mutability in `const` items
const G: ReentrantLock<i32> = ReentrantLock::new(0);
//~^ WARN interior mutability in `const` items
const H: RwLock<i32> = RwLock::new(0);
//~^ WARN interior mutability in `const` items
const I: AtomicBool = AtomicBool::new(false);
//~^ WARN interior mutability in `const` items
const J: AtomicPtr<i32> = AtomicPtr::new(std::ptr::null_mut());
//~^ WARN interior mutability in `const` items
const K: AtomicI32 = AtomicI32::new(0);
//~^ WARN interior mutability in `const` items
const L: AtomicU32 = AtomicU32::new(0);
//~^ WARN interior mutability in `const` items

pub(crate) const X: Once = Once::new();
//~^ WARN interior mutability in `const` items

fn main() {
    const Z: Once = Once::new();
    //~^ WARN interior mutability in `const` items
}

struct S;
impl S {
    const Z: Once = Once::new(); // not a const-item
}
