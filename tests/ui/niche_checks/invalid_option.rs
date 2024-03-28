//@ run-fail
//@ ignore-wasm32-bare: No panic messages
//@ compile-flags: -C debug-assertions -Zmir-opt-level=0
//@ error-pattern: occupied niche: found 2 but must be in 0..=1

fn main() {
    unsafe {
        std::mem::transmute::<(u8, u8), Option<u8>>((2, 0));
    }
}
