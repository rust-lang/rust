//! Regression test for <https://github.com/rust-lang/rust/issues/5062>.

fn main() {
    None; //~ ERROR type annotations needed [E0282]
}
