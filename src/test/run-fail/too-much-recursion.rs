// error-pattern:ran out of stack

// Test that the task fails after hiting the recursion limit

fn main() {
    main();
}