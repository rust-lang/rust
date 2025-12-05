//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: hint::assert_unchecked must never be called when the condition is false

fn main() {
    unsafe {
        std::hint::assert_unchecked(false);
    }
}
