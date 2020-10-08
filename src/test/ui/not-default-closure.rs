// Check that closures do not implement `Default` if their environment is not empty.

fn test_default<F: Default + Fn()>(_: F) {
    let f = F::default();
    f();
}

fn main() {
    let a = "Bob";
    let hello = move || {
        println!("Hello {}", a);
    };

    test_default(hello); //~ ERROR the trait bound
}
