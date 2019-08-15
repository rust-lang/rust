// aux-build:issue-63226.rs
// compile-flags:--extern issue_63226
// edition:2018
// build-pass

use issue_63226::VTable;

static ICE_ICE:&'static VTable=VTable::vtable();

fn main() {}
