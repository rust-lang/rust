// run-fail
// compile-flags: -Copt-level=3 -Cdebug-assertions=yes
// error-pattern: unsafe precondition(s) violated: slice::from_raw_parts
// ignore-debug
// ignore-wasm32-bare no panic messages

fn main() {
    unsafe {
        let _s: &[u64] = std::slice::from_raw_parts(1usize as *const u64, 0);
    }
}
