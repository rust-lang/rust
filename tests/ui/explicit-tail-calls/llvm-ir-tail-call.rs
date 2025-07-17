//@ compile-flags: -O
//@ run-pass
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

// A deep recursive function that uses explicit tail calls
// This will cause stack overflow without tail call optimization
fn deep_recursion(n: u32) -> u32 {
    match n {
        0 => 0,
        _ => become deep_recursion(n - 1)
    }
}

// A deep recursive function without explicit tail calls
// This will overflow the stack for large values
fn deep_recursion_no_tail(n: u32) -> u32 {
    match n {
        0 => 0,
        _ => deep_recursion_no_tail(n - 1)
    }
}

fn main() {
    // Verify correctness for small values
    assert_eq!(deep_recursion(10), 0);
    assert_eq!(deep_recursion_no_tail(10), 0);
    
    // This will succeed only if tail call optimization is working
    // It would overflow the stack otherwise
    println!("Starting deep recursion with 50,000 calls");
    let result = deep_recursion(50_000);
    assert_eq!(result, 0);
    println!("Successfully completed 50,000 recursive calls with tail call optimization");
}
