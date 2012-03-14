// xfail-fast
// aux-build:native_lib.rs

// The purpose of this test is to check that we can
// successfully (and safely) invoke external, cdecl
// functions from outside the crate.

use native_lib;

fn main() {
    let foo = native_lib::rustrt::last_os_error();
}
