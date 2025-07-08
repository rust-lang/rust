//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ check-run-results

fn main() {
    unsafe {
        std::hint::unreachable_unchecked();
    }
}
