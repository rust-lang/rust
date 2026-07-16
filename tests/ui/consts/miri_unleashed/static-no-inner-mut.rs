//@ stderr-per-bitwidth
//@ compile-flags: -Zunleash-the-miri-inside-of-you

// All "inner" allocations that come with a `static` are interned immutably. This means it is
// crucial that we do not accept any form of (interior) mutability there.
use std::sync::atomic::*;

static REF: &AtomicI32 = &AtomicI32::new(42);
//~^ ERROR `UnsafeCell` in read-only memory

static REFMUT: &mut i32 = &mut 0;
//~^ ERROR mutable reference or box pointing to read-only memory

// Different way of writing this that avoids promotion.
static REF2: &AtomicI32 = {let x = AtomicI32::new(42); &{x}};
//~^ ERROR `UnsafeCell` in read-only memory
static REFMUT2: &mut i32 = {let mut x = 0; &mut {x}};
//~^ ERROR mutable reference or box pointing to read-only memory

// This one is obvious, since it is non-Sync. (It also suppresses the other errors, so it is
// commented out.)
// static RAW: *const AtomicI32 = &AtomicI32::new(42);

struct SyncPtr<T> { x : *const T }
unsafe impl<T> Sync for SyncPtr<T> {}

// All of these pass the lifetime checks because of the "tail expression" / "outer scope" rule.
// (This relies on `SyncPtr` being a curly brace struct.)
// Then they get interned immutably, which is not great. See
// <https://github.com/rust-lang/rust/pull/128543> for why we accept such code.
static RAW_SYNC: SyncPtr<AtomicI32> = SyncPtr { x: &AtomicI32::new(42) };

// With mutable references at least, we can detect this and error.
static RAW_MUT_CAST: SyncPtr<i32> = SyncPtr { x : &mut 42 as *mut _ as *const _ };
//~^ ERROR mutable pointer in final value

static RAW_MUT_COERCE: SyncPtr<i32> = SyncPtr { x: &mut 0 };
//~^ ERROR mutable pointer in final value

fn main() {}

//~? WARN skipping const checks
