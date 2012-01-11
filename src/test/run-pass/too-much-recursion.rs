// error-pattern:ran out of stack

// Test that the task fails after hiting the recursion limit, but
// that it doesn't bring down the whole proc

fn main() {
    task::spawn {||
        task::unsupervise();
        fn f() { f() };
        f();
    };
}