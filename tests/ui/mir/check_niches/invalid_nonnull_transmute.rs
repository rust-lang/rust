// run-fail
// ignore-wasm32-bare: No panic messages
// compile-flags: -C debug-assertions -Zmir-opt-level=0
// error-pattern: occupied niche: found 0x0 but must be in 0x1..=0xffffffff

fn main() {
    unsafe {
        std::mem::transmute::<*const u8, std::ptr::NonNull<u8>>(std::ptr::null());
    }
}
