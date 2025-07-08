//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ check-run-results

fn main() {
    unsafe {
        char::from_u32_unchecked(0xD801);
    }
}
