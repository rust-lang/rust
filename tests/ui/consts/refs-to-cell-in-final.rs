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

// This one does not get promoted because of `Drop`, and then enters interesting codepaths because
// as a value it has no interior mutability, but as a type it does. See
// <https://github.com/rust-lang/rust/issues/121610>. Value-based reasoning for interior mutability
// is questionable (https://github.com/rust-lang/unsafe-code-guidelines/issues/493) so for now we
// reject this, though not with a great error message.
pub enum JsValue {
    Undefined,
    Object(Cell<bool>),
}
impl Drop for JsValue {
    fn drop(&mut self) {}
}
const UNDEFINED: &JsValue = &JsValue::Undefined;
//~^ERROR: mutable pointer in final value of constant

// In contrast, this one works since it is being promoted.
const NONE: &'static Option<Cell<i32>> = &None;
// Making it clear that this is promotion, not "outer scope".
const NONE_EXPLICIT_PROMOTED: &'static Option<Cell<i32>> = {
    let x = &None;
    x
};

fn main() {}
