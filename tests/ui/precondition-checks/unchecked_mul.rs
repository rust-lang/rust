//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ check-run-results

fn main() {
    unsafe {
        2u8.unchecked_add(u8::MAX);
    }
}
