//@ run-pass
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

// A function that causes deep recursion - counts down from n to 0
// Without tail calls, this would overflow the stack for large n values
fn countdown(n: u32) -> u32 {
    match n {
        0 => 0,
        _ => become countdown(n - 1)
    }
}

// Same function but without tail call optimization
fn countdown_no_tail(n: u32) -> u32 {
    match n {
        0 => 0,
        _ => countdown_no_tail(n - 1)
    }
}

// This test is specifically designed to verify tail call optimization
// We use an extremely large recursion depth (500,000) that would 
// absolutely overflow the stack without tail call optimization
fn main() {
    // Small test to verify correctness
    assert_eq!(countdown(10), 0);
    
    // Regular recursion would overflow here (500,000 stack frames)
    // Only works if tail call optimization is actually happening
    let result = countdown(500_000);
    assert_eq!(result, 0);
    println!("Successfully completed 500,000 recursive calls with tail call optimization");
    
    // We can't test the non-tail version with a large number as it would crash,
    // but we can verify it works for small values
    assert_eq!(countdown_no_tail(10), 0);
}