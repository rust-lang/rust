//! regression test for <https://github.com/rust-lang/rust/issues/3109>
//@ run-pass
pub fn main() {
    println!("{:?}", ("hi there!", "you"));
}
