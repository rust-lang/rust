//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ check-run-results: unsafe precondition(s) violated: `ascii::Char::digit_unchecked`

#![feature(ascii_char)]

fn main() {
    unsafe {
        std::ascii::Char::digit_unchecked(b'a');
    }
}
