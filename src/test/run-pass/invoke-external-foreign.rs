// xfail-fast
// aux-build:foreign_lib.rs

// The purpose of this test is to check that we can
// successfully (and safely) invoke external, cdecl
// functions from outside the crate.

extern mod foreign_lib;

fn main() {
    let foo = foreign_lib::rustrt::last_os_error();
}
