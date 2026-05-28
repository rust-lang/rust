// Test internal const fn feature gate.

#[rustc_const_unstable(feature="fzzzzzt")]
//~^ ERROR stability attributes may not be used outside
//~| ERROR missing 'issue'
pub const fn bazinga() {}

fn main() {
}
