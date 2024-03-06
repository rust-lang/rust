//@ run-pass
//@ compile-flags: -C debug-assertions

fn main() {
    let ptr = 1 as *const u16;
    unsafe {
        let _ = *ptr;
    }
}
