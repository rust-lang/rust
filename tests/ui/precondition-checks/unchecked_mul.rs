//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: u8::unchecked_add cannot overflow

fn main() {
    unsafe {
        2u8.unchecked_add(u8::MAX);
    }
}
