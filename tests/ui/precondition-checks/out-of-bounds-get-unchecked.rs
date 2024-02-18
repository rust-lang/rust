//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=yes
//@ error-pattern: unsafe precondition(s) violated: hint::assert_unchecked
//@ ignore-debug
//@ ignore-wasm32-bare no panic messages

fn main() {
    unsafe {
        let sli: &[u8] = &[0];
        sli.get_unchecked(1);
    }
}
