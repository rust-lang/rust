//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: as_ascii_unchecked requires that the
//@ revisions: char str

#![feature(ascii_char)]

use std::ascii::Char;

fn main() {
    unsafe {
        #[cfg(char)]
        let _c: Char = '🦀'.as_ascii_unchecked();
        #[cfg(str)]
        let _c: &[Char] = "🦀".as_ascii_unchecked();
    }
}
