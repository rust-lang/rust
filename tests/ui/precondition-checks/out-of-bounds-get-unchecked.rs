//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=yes
//@ error-pattern: slice::get_unchecked requires
//@ ignore-debug

fn main() {
    unsafe {
        let sli: &[u8] = &[0];
        sli.get_unchecked(1);
    }
}
