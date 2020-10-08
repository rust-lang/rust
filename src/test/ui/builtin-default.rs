// run-pass
// Test that `Default` is correctly implemented for builtin types.

fn test_default<F: Default + Fn()>(_: F) {
    let f = F::default();
    f();
}

fn foo() { }

fn main() {
    test_default(foo);
    test_default(|| ());
}
