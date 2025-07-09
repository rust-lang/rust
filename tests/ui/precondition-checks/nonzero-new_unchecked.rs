//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ check-run-results

fn main() {
    unsafe {
        std::num::NonZeroU8::new_unchecked(0);
    }
}
