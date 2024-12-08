//@ aux-build:issue-63226.rs
//@ compile-flags:--extern issue_63226
//@ edition:2018
//@ build-pass
// A regression test for issue #63226.
// Checks if `const fn` is marked as reachable.

use issue_63226::VTable;

static ICE_ICE:&'static VTable=VTable::vtable();
static MORE_ICE:&'static VTable=VTable::VTABLE2;
static MORE_ICE3:&'static VTable=issue_63226::VTABLE3;

fn main() {}
