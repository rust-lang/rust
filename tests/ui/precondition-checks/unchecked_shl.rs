//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: u8::unchecked_shl cannot overflow

#![feature(unchecked_shifts)]

fn main() {
    unsafe {
        0u8.unchecked_shl(u8::BITS);
    }
}
