//@ run-pass

#![allow(unpredictable_function_pointer_comparisons)]

fn main() {
    // We shouldn't promote this
    let _ = &(main as fn() == main as fn());
    // Also check nested case
    let _ = &(&(main as fn()) == &(main as fn()));
}
