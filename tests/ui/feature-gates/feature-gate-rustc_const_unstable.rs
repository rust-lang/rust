// Test internal const fn feature gate.

#[rustc_const_unstable(feature="fzzzzzt")] //~ ERROR stability attributes may not be used outside
pub const fn bazinga() {}

fn main() {
}
