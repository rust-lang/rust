//@ run-fail
//@ ignore-wasm32-bare: No panic messages
//@ compile-flags: -Zmir-opt-level=0 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: occupied niche: found 2 but must be in 0..=1

fn main() {
    unsafe {
        std::mem::transmute::<(u8, u8), Option<u8>>((2, 0));
    }
}
