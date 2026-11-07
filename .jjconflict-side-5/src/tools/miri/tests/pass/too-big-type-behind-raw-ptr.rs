//! Regression test for <https://github.com/rust-lang/rust/issues/157654#issuecomment-4679655886>:
//! the type behind a raw pointer should never have its layout computed.

//@compile-flags: -Zmiri-permissive-provenance

const PTR_BITS_MINUS_1: usize = std::mem::size_of::<*const ()>() * 8 - 1;

fn main() {
    std::hint::black_box(0 as *const [u64; 1 << PTR_BITS_MINUS_1]);
}
