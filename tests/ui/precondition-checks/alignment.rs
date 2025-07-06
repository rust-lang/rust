//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: Alignment::new_unchecked requires

#![feature(ptr_alignment_type)]

fn main() {
    unsafe {
        std::ptr::Alignment::new_unchecked(0);
    }
}
