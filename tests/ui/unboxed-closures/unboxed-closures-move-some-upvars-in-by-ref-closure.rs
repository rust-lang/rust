// run-pass
// Test that in a by-ref once closure we move some variables even as
// we capture others by mutable reference.

fn call<F>(f: F) where F : FnOnce() {
    f();
}

fn main() {
    let mut x = vec![format!("Hello")];
    let y = vec![format!("World")];
    call(|| {
        // Here: `x` must be captured with a mutable reference in
        // order for us to append on it, and `y` must be captured by
        // value.
        for item in y {
            x.push(item);
        }
    });
    assert_eq!(x.len(), 2);
    assert_eq!(&*x[0], "Hello");
    assert_eq!(&*x[1], "World");
}
