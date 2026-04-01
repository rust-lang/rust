//! regression test for <https://github.com/rust-lang/rust/issues/38987>
//@ run-pass
fn main() {
    let _ = -0x8000_0000_0000_0000_0000_0000_0000_0000i128;
}
