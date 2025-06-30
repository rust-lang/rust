//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: u8::unchecked_sub cannot overflow

fn main() {
    unsafe {
        0u8.unchecked_sub(1u8);
    }
}
