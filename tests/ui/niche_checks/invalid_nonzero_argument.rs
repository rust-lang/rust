//@ run-fail
//@ ignore-wasm32-bare: No panic messages
//@ compile-flags: -Zmir-opt-level=0 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: occupied niche: found 0 but must be in 1..=255

fn main() {
    let mut bad = std::num::NonZeroU8::new(1u8).unwrap();
    unsafe {
        std::ptr::write_bytes(&mut bad, 0u8, 1usize);
    }
    func(bad);
}

fn func<T>(_t: T) {}
