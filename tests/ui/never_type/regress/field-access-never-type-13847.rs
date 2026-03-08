//! Regression test for https://github.com/rust-lang/rust/issues/13847

fn main() {
    return.is_failure //~ ERROR no field `is_failure` on type `!`
}
