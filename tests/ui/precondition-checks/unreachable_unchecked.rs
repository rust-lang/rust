//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ check-run-results: unsafe precondition(s) violated: hint::unreachable_unchecked must never

fn main() {
    unsafe {
        std::hint::unreachable_unchecked();
    }
}
