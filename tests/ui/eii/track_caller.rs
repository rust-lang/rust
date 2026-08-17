//@ run-pass
//@ ignore-backends: gcc
//@ ignore-windows
// Tests that `#[track_caller]` on an EII declaration is threaded through both
// the default impl (no override) and an explicit override (which does not
// repeat `#[track_caller]` — it is derived during codegen so the shim ABI matches).

#![feature(extern_item_impls)]

use std::panic::Location;
use std::sync::atomic::{AtomicU32, Ordering};

static LAST_LINE: AtomicU32 = AtomicU32::new(0);

#[track_caller]
#[eii]
fn decl_default(_x: u64) {
    LAST_LINE.store(Location::caller().line(), Ordering::SeqCst);
}

#[track_caller]
#[eii]
fn decl_override(_x: u64);

#[decl_override]
fn explicit_override(_x: u64) {
    LAST_LINE.store(Location::caller().line(), Ordering::SeqCst);
}

fn main() {
    decl_default(1);
    assert_eq!(LAST_LINE.load(Ordering::SeqCst), line!() - 1);

    decl_override(2);
    assert_eq!(LAST_LINE.load(Ordering::SeqCst), line!() - 1);
}
