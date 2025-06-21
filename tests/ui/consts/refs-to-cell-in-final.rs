use std::cell::*;

struct SyncPtr<T> {
    x: *const T,
}
unsafe impl<T> Sync for SyncPtr<T> {}

// These pass the lifetime checks because of the "tail expression" / "outer scope" rule.
// (This relies on `SyncPtr` being a curly brace struct.)
// However, we intern the inner memory as read-only.
// The resulting constant would pass all validation checks, so it is crucial that this gets rejected
// by static const checks!
static RAW_SYNC_S: SyncPtr<Cell<i32>> = SyncPtr { x: &Cell::new(42) };
//~^ ERROR: interior mutable shared borrows of lifetime-extended temporaries
const RAW_SYNC_C: SyncPtr<Cell<i32>> = SyncPtr { x: &Cell::new(42) };
//~^ ERROR: interior mutable shared borrows of lifetime-extended temporaries

// This one does not get promoted because of `Drop`, and then enters interesting codepaths because
// as a value it has no interior mutability, but as a type it does. See
// <https://github.com/rust-lang/rust/issues/121610>. Value-based reasoning for interior mutability
// is questionable (https://github.com/rust-lang/unsafe-code-guidelines/issues/493) but we've
// done it since Rust 1.0 so we can't stop now.
pub enum JsValue {
    Undefined,
    Object(Cell<bool>),
}
impl Drop for JsValue {
    fn drop(&mut self) {}
}
const UNDEFINED: &JsValue = &JsValue::Undefined;

// Here's a variant of the above that uses promotion instead of the "outer scope" rule.
const NONE: &'static Option<Cell<i32>> = &None;
// Making it clear that this is promotion, not "outer scope".
const NONE_EXPLICIT_PROMOTED: &'static Option<Cell<i32>> = {
    let x = &None;
    x
};

// Not okay, since we are borrowing something with interior mutability.
const INTERIOR_MUT_VARIANT: &Option<UnsafeCell<bool>> = &{
    //~^ERROR: interior mutable shared borrows of lifetime-extended temporaries
    let mut x = None;
    assert!(x.is_none());
    x = Some(UnsafeCell::new(false));
    x
};

fn main() {}
