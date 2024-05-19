//@ run-pass
// Check that closures implement `Copy`.

fn call<T, F: FnOnce() -> T>(f: F) -> T { f() }

fn main() {
    let a = 5;
    let hello = || {
        println!("Hello {}", a);
        a
    };

    assert_eq!(5, call(hello.clone()));
    assert_eq!(5, call(hello));
    assert_eq!(5, call(hello));
}
