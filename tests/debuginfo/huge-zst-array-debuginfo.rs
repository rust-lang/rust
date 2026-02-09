// Verify that the compiler doesn't crash when generating debug info
// for a huge array of zero-sized types.
// See: https://github.com/rust-lang/rust/issues/34127

//@ compile-flags:-g

fn main() {
    let _a = [(); 1 << 63];
}
