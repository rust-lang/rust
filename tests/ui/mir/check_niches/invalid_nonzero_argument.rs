// run-fail
// ignore-wasm32-bare: No panic messages
// compile-flags: -C debug-assertions -Zmir-opt-level=0
// error-pattern: occupied niche: found 0x0 but must be in 0x1..=0xff

fn main() {
    let mut bad = std::num::NonZeroU8::new(1u8).unwrap();
    unsafe {
        std::ptr::write_bytes(&mut bad, 0u8, 1usize);
    }
    func(bad);
}

fn func<T>(_t: T) {}
