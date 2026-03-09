// Verify that E0434 gives context-appropriate help messages.
// Regression test for https://github.com/rust-lang/rust/issues/153363

// Case 1: Nested fn item - closure conversion is possible.
fn nested_fn(x: String) {
    fn inner() {
        let _ = x; //~ ERROR can't capture dynamic environment in a fn item
    }
}

// Case 2: Nested fn returning a closure - closure conversion is possible.
fn nested_fn_with_closure(x: String) {
    fn inner() -> impl FnOnce() -> () {
        move || { let _ = x; } //~ ERROR can't capture dynamic environment in a fn item
    }
}

// Case 3: Trait impl method - closure conversion is impossible.
trait T {
    fn method() -> impl FnOnce() -> ();
}

fn trait_impl(x: String) {
    struct S;
    impl T for S {
        fn method() -> impl FnOnce() -> () {
            move || { let _ = x; } //~ ERROR can't capture dynamic environment in a fn item
        }
    }
}

// Case 4: Inherent impl method - closure conversion is impossible.
fn inherent_impl(x: String) {
    struct S;
    impl S {
        fn method() {
            let _ = x; //~ ERROR can't capture dynamic environment in a fn item
        }
    }
}

fn main() {}
