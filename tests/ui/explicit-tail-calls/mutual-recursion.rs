//@ run-pass
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

// This is a classic example of mutual recursion:
// even(n) calls odd(n-1), and odd(n) calls even(n-1)
// Without tail calls, this would quickly overflow the stack for large inputs

// Check if a number is even using mutual recursion
fn is_even(n: u64) -> bool {
    match n {
        0 => true,
        _ => become is_odd(n - 1)
    }
}

// Check if a number is odd using mutual recursion
fn is_odd(n: u64) -> bool {
    match n {
        0 => false,
        _ => become is_even(n - 1)
    }
}

// Versions without tail calls for comparison
fn is_even_no_tail(n: u64) -> bool {
    match n {
        0 => true,
        _ => is_odd_no_tail(n - 1)
    }
}

fn is_odd_no_tail(n: u64) -> bool {
    match n {
        0 => false,
        _ => is_even_no_tail(n - 1)
    }
}

fn main() {
    // Verify correctness for small values
    assert_eq!(is_even(0), true);
    assert_eq!(is_odd(0), false);
    assert_eq!(is_even(1), false);
    assert_eq!(is_odd(1), true);
    assert_eq!(is_even(10), true);
    assert_eq!(is_odd(10), false);
    
    // Test with an extremely large number that would definitely overflow the stack
    // without tail call optimization - each call creates 2 stack frames (alternating between functions)
    // so 100,000 would create 200,000 stack frames total
    assert_eq!(is_even(100_000), true);
    assert_eq!(is_odd(100_000), false);
    assert_eq!(is_even(100_001), false);
    assert_eq!(is_odd(100_001), true);
    
    println!("Deep mutual recursion test passed with 100,000 alternating recursive calls!");
    
    // Verify non-tail versions work for small values
    assert_eq!(is_even_no_tail(10), true);
    assert_eq!(is_odd_no_tail(10), false);
    // But would overflow for large values (not tested to avoid crashing)
}