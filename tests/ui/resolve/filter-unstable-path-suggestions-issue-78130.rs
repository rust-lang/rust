// Regression test for issue https://github.com/rust-lang/rust/issues/78130
// Rust should not suggest unstable paths in the error message.

pub fn main() {
    let _ = assert_zero_valid::<()>();
                    //~^ ERROR cannot find function `assert_zero_valid` in this scope
}
