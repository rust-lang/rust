// run-pass
// ignore-wasm32-bare: No panic messages
// compile-flags: -C debug-assertions

fn main() {
    unsafe {
        std::mem::transmute::<u8, bool>(0);
    }
    unsafe {
        std::mem::transmute::<u8, bool>(1);
    }
}
