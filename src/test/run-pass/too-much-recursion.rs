// xfail-win32
// error-pattern:ran out of stack

// Test that the task fails after hitting the recursion limit, but
// that it doesn't bring down the whole proc

fn main() {
    do task::spawn_unlinked {
        fn f() { f() };
        f();
    };
}
