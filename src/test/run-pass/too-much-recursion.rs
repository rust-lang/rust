// xfail-win32
// error-pattern:ran out of stack

// Test that the task fails after hiting the recursion limit, but
// that it doesn't bring down the whole proc

fn main() {
    let builder = task::builder();
    task::unsupervise(builder);
    task::run(builder) {||
        fn f() { f() };
        f();
    };
}