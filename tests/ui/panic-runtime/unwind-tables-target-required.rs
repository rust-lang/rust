// Tests that the compiler errors if the user tries to turn off unwind tables
// when they are required.
//
//@only-target-x86_64-pc-windows-msvc
//@compile-flags: -C force-unwind-tables=no
//
// dont-check-compiler-stderr
// ignore-tidy-linelength
//@error-in-other-file: target requires unwind tables, they cannot be disabled with `-C force-unwind-tables=no`

pub fn main() {
}
