//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ check-run-results

#![feature(unchecked_shifts)]

fn main() {
    unsafe {
        0u8.unchecked_shr(u8::BITS);
    }
}
