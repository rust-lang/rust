//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: NonZero::new_unchecked requires

fn main() {
    unsafe {
        std::num::NonZeroU8::new_unchecked(0);
    }
}
