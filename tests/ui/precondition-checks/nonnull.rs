//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ check-run-results

fn main() {
    unsafe {
        std::ptr::NonNull::new_unchecked(std::ptr::null_mut::<u8>());
    }
}
