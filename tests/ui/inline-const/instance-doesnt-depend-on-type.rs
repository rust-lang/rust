//@ check-pass
// issue: 114660

fn main() {
    const { core::mem::transmute::<u8, u8> };
    // Don't resolve the instance of this inline constant to be an intrinsic,
    // even if the type of the constant is `extern "rust-intrinsic" fn(u8) -> u8`.
}
