//! regression test for <https://github.com/rust-lang/rust/issues/27268>
//@ run-pass
fn main() {
    const _C: &'static dyn Fn() = &|| {};
}
