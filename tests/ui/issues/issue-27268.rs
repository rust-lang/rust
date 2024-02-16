//@ run-pass
fn main() {
    const _C: &'static dyn Fn() = &||{};
}
