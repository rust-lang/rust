//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: hint::unreachable_unchecked must never be reached

fn main() {
    unsafe {
        std::hint::unreachable_unchecked();
    }
}
