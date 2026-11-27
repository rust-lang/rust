//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: NonNull::new_unchecked requires

fn main() {
    unsafe {
        std::ptr::NonNull::new_unchecked(std::ptr::null_mut::<u8>());
    }
}
