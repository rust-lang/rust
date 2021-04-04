// Tests that the compiler errors if the user tries to turn off unwind tables
// when they are required.
//
// only-x86_64-windows-msvc
// compile-flags: -C force-unwind-tables=no
// ignore-tidy-linelength
//
// error-pattern: target requires unwind tables, they cannot be disabled with `-C force-unwind-tables=no`.

pub fn main() {
}
