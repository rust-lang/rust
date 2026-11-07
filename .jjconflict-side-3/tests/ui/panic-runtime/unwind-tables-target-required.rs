// Tests that the compiler errors if the user tries to turn off unwind tables
// when they are required.
//
//@ only-x86_64-pc-windows-msvc
//@ compile-flags: -C force-unwind-tables=no
//
//@ dont-check-compiler-stderr

pub fn main() {
}

//~? ERROR target requires unwind tables, they cannot be disabled with `-C force-unwind-tables=no`
