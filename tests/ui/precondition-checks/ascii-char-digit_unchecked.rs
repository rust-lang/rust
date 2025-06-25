//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: `ascii::Char::digit_unchecked` input cannot exceed 9

#![feature(ascii_char)]

fn main() {
    unsafe {
        std::ascii::Char::digit_unchecked(b'a');
    }
}
