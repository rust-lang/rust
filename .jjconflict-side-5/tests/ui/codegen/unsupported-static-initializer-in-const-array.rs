//! LLVM error with unsupported expression in static
//! initializer for const pointer in array on macOS.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/89225>.

//@ build-pass
//@ compile-flags: -C opt-level=3

const fn make() -> (i32, i32, *const i32) {
    const V: i32 = 123;
    &V as *const i32;
    (0, 0, &V)
}

fn main() {
    let arr = [make(); 32];
    println!("{}", arr[0].0);
}
