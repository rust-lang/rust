// run-fail
// ignore-wasm32-bare: No panic messages
// compile-flags: -C debug-assertions
// error-pattern: occupied niche: found 0x2 but must be in 0x0..=0x1

fn main() {
    unsafe {
        std::mem::transmute::<u8, bool>(2);
    }
}
