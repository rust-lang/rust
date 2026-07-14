//! Regression test for https://github.com/rust-lang/rust/issues/46311
fn main() {
    'break: loop { //~ ERROR labels cannot use keyword names
    }
}
