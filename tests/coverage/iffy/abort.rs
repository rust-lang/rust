#![allow(unused_assignments)]

extern "C" fn might_abort(should_abort: bool) {
    if should_abort {
        println!("aborting...");
        panic!("panics and aborts");
    } else {
        println!("Don't Panic");
    }
}

#[rustfmt::skip]
fn main() -> Result<(), u8> {
    let mut countdown = 10;
    while countdown > 0 {
        if countdown < 5 {
            might_abort(false);
        }
        // See discussion (below the `Notes` section) on coverage results for the closing brace.
        if countdown < 5 { might_abort(false); } // Counts for different regions on one line.
        // For the following example, the closing brace is the last character on the line.
        // This shows the character after the closing brace is highlighted, even if that next
        // character is a newline.
        if countdown < 5 { might_abort(false); }
        countdown -= 1;
    }
    Ok(())
}

// Notes:
//   1. Compare this program and its coverage results to those of the similar tests
//      `panic_unwind.rs` and `try_error_result.rs`.
//   2. This test confirms the coverage generated when a program includes `UnwindAction::Terminate`.
//   3. The test does not invoke the abort. By executing to a successful completion, the coverage
//      results show where the program did and did not execute.
//   4. If the program actually aborted, the coverage counters would not be saved (which "works as
//      intended"). Coverage results would show no executed coverage regions.
//   6. If `should_abort` is `true` and the program aborts, the program exits with a `132` status
//      (on Linux at least).

/*

Expect the following coverage results:

```text
    16|     11|    while countdown > 0 {
    17|     10|        if countdown < 5 {
    18|      4|            might_abort(false);
    19|      6|        }
```

This is actually correct.

The condition `countdown < 5` executed 10 times (10 loop iterations).

It evaluated to `true` 4 times, and executed the `might_abort()` call.

It skipped the body of the `might_abort()` call 6 times. If an `if` does not include an explicit
`else`, the coverage implementation injects a counter, at the character immediately after the `if`s
closing brace, to count the "implicit" `else`. This is the only way to capture the coverage of the
non-true condition.

As another example of why this is important, say the condition was `countdown < 50`, which is always
`true`. In that case, we wouldn't have a test for what happens if `might_abort()` is not called.
The closing brace would have a count of `0`, highlighting the missed coverage.
*/
