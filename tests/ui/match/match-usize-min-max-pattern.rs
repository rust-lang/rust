//! regression test for <https://github.com/rust-lang/rust/issues/37686>
//@ check-pass
fn main() {
    match (0, 0) {
        (usize::MIN, usize::MAX) => {}
        _ => {}
    }
}
