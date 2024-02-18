#![feature(const_refs_to_cell)]

use std::cell::*;

struct SyncPtr<T> { x : *const T }
unsafe impl<T> Sync for SyncPtr<T> {}

// These pass the lifetime checks because of the "tail expression" / "outer scope" rule.
// (This relies on `SyncPtr` being a curly brace struct.)
// However, we intern the inner memory as read-only.
// The resulting constant would pass all validation checks, so it is crucial that this gets rejected
// by static const checks!
static RAW_SYNC_S: SyncPtr<Cell<i32>> = SyncPtr { x: &Cell::new(42) };
//~^ ERROR: cannot refer to interior mutable data
const RAW_SYNC_C: SyncPtr<Cell<i32>> = SyncPtr { x: &Cell::new(42) };
//~^ ERROR: cannot refer to interior mutable data

fn main() {}
