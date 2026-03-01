//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: Alignment::new_unchecked requires

fn main() {
    unsafe {
        std::mem::Alignment::new_unchecked(0);
    }
}
