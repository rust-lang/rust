//@ stderr-per-bitwidth
//@ compile-flags: -Zunleash-the-miri-inside-of-you
#![feature(const_refs_to_cell, const_mut_refs)]
// All "inner" allocations that come with a `static` are interned immutably. This means it is
// crucial that we do not accept any form of (interior) mutability there.

use std::sync::atomic::*;

static REF: &AtomicI32 = &AtomicI32::new(42); //~ERROR mutable pointer in final value
static REFMUT: &mut i32 = &mut 0; //~ERROR mutable pointer in final value

// Different way of writing this that avoids promotion.
static REF2: &AtomicI32 = {let x = AtomicI32::new(42); &{x}}; //~ERROR mutable pointer in final value
static REFMUT2: &mut i32 = {let mut x = 0; &mut {x}}; //~ERROR mutable pointer in final value

// This one is obvious, since it is non-Sync. (It also suppresses the other errors, so it is
// commented out.)
// static RAW: *const AtomicI32 = &AtomicI32::new(42);

struct SyncPtr<T> { x : *const T }
unsafe impl<T> Sync for SyncPtr<T> {}

// All of these pass the lifetime checks because of the "tail expression" / "outer scope" rule.
// (This relies on `SyncPtr` being a curly brace struct.)
// Then they get interned immutably, which is not great.
// `mut_ref_in_final.rs` and `std/cell.rs` ensure that we don't accept this even with the feature
// fate, but for unleashed Miri there's not really any way we can reject them: it's just
// non-dangling raw pointers.
static RAW_SYNC: SyncPtr<AtomicI32> = SyncPtr { x: &AtomicI32::new(42) };
//~^ ERROR mutable pointer in final value
static RAW_MUT_CAST: SyncPtr<i32> = SyncPtr { x : &mut 42 as *mut _ as *const _ };
//~^ ERROR mutable pointer in final value
static RAW_MUT_COERCE: SyncPtr<i32> = SyncPtr { x: &mut 0 };
//~^ ERROR mutable pointer in final value

fn main() {}
