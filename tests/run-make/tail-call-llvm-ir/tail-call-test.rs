#![feature(explicit_tail_calls)]

// A recursive function that uses explicit tail calls with 'become'
// This should generate LLVM IR with "tail call" instructions
pub fn with_tail(n: u32) -> u32 {
    if n == 0 { 0 } else { become with_tail(n - 1) }
}

// A similar recursive function but without 'become'
// This should NOT generate tail call instructions by default
pub fn no_tail(n: u32) -> u32 {
    if n == 0 { 0 } else { no_tail(n - 1) }
}

// Mutual recursion with 'become'
// Both functions should generate tail call instructions
pub fn even_with_tail(n: u32) -> bool {
    match n {
        0 => true,
        _ => become odd_with_tail(n - 1),
    }
}

pub fn odd_with_tail(n: u32) -> bool {
    match n {
        0 => false,
        _ => become even_with_tail(n - 1),
    }
}

// For comparison: mutual recursion without 'become'
// These should NOT generate tail call instructions
pub fn even_no_tail(n: u32) -> bool {
    match n {
        0 => true,
        _ => odd_no_tail(n - 1),
    }
}

pub fn odd_no_tail(n: u32) -> bool {
    match n {
        0 => false,
        _ => even_no_tail(n - 1),
    }
}

fn main() {
    // Actually use all the functions to ensure they're not optimized away
    let with_tail_result = with_tail(5);
    let no_tail_result = no_tail(5);
    let even_with_tail_result = even_with_tail(10);
    let odd_with_tail_result = odd_with_tail(10);
    let even_no_tail_result = even_no_tail(10);
    let odd_no_tail_result = odd_no_tail(10);

    println!(
        "Results: {} {} {} {} {} {}",
        with_tail_result,
        no_tail_result,
        even_with_tail_result,
        odd_with_tail_result,
        even_no_tail_result,
        odd_no_tail_result
    );
}
