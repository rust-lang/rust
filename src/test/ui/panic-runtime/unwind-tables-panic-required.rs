// Tests that the compiler errors if the user tries to turn off unwind tables
// when they are required.
//
// dont-check-compiler-stderr
// compile-flags: -C panic=unwind -C force-unwind-tables=no
// ignore-tidy-linelength
//
// error-pattern: panic=unwind requires unwind tables, they cannot be disabled with `-C force-unwind-tables=no`.

pub fn main() {
}
