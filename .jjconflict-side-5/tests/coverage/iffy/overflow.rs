#![allow(unused_assignments)]
//@ compile-flags: -Coverflow-checks=yes
//@ failure-status: 101

fn might_overflow(to_add: u32) -> u32 {
    if to_add > 5 {
        println!("this will probably overflow");
    }
    let add_to = u32::MAX - 5;
    println!("does {} + {} overflow?", add_to, to_add);
    let result = to_add + add_to;
    println!("continuing after overflow check");
    result
}

fn main() -> Result<(), u8> {
    let mut countdown = 10;
    while countdown > 0 {
        if countdown == 1 {
            let result = might_overflow(10);
            println!("Result: {}", result);
        } else if countdown < 5 {
            let result = might_overflow(1);
            println!("Result: {}", result);
        }
        countdown -= 1;
    }
    Ok(())
}

// Notes:
//   1. Compare this program and its coverage results to those of the very similar test `assert.rs`,
//      and similar tests `panic_unwind.rs`, abort.rs` and `try_error_result.rs`.
//   2. This test confirms the coverage generated when a program passes or fails a
//      compiler-generated `TerminatorKind::Assert` (based on an overflow check, in this case).
//   3. Similar to how the coverage instrumentation handles `TerminatorKind::Call`,
//      compiler-generated assertion failures are assumed to be a symptom of a program bug, not
//      expected behavior. To simplify the coverage graphs and keep instrumented programs as
//      small and fast as possible, `Assert` terminators are assumed to always succeed, and
//      therefore are considered "non-branching" terminators. So, an `Assert` terminator does not
//      get its own coverage counter.
//   4. After an unhandled panic or failed Assert, coverage results may not always be intuitive.
//      In this test, the final count for the statements after the `if` block in `might_overflow()`
//      is 4, even though the lines after `to_add + add_to` were executed only 3 times. Depending
//      on the MIR graph and the structure of the code, this count could have been 3 (which might
//      have been valid for the overflowed add `+`, but should have been 4 for the lines before
//      the overflow. The reason for this potential uncertainty is, a `CounterKind` is incremented
//      via StatementKind::Counter at the end of the block, but (as in the case in this test),
//      a CounterKind::Expression is always evaluated. In this case, the expression was based on
//      a `Counter` incremented as part of the evaluation of the `if` expression, which was
//      executed, and counted, 4 times, before reaching the overflow add.

// If the program did not overflow, the coverage for `might_overflow()` would look like this:
//
//     4|       |fn might_overflow(to_add: u32) -> u32 {
//     5|      4|    if to_add > 5 {
//     6|      0|        println!("this will probably overflow");
//     7|      4|    }
//     8|      4|    let add_to = u32::MAX - 5;
//     9|      4|    println!("does {} + {} overflow?", add_to, to_add);
//    10|      4|    let result = to_add + add_to;
//    11|      4|    println!("continuing after overflow check");
//    12|      4|    result
//    13|      4|}
