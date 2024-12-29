//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ check-run-results: unsafe precondition(s) violated: u8::unchecked_add cannot overflow

fn main() {
    unsafe {
        1u8.unchecked_add(u8::MAX);
    }
}
