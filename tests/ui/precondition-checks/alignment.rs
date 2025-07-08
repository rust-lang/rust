//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ check-run-results

#![feature(ptr_alignment_type)]

fn main() {
    unsafe {
        std::ptr::Alignment::new_unchecked(0);
    }
}
