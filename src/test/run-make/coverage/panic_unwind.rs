#![allow(unused_assignments)]
// expect-exit-status-101

fn might_panic(should_panic: bool) {
    if should_panic {
        println!("panicking...");
        panic!("panics");
    } else {
        println!("Don't Panic");
    }
}

fn main() -> Result<(), u8> {
    let mut countdown = 10;
    while countdown > 0 {
        if countdown == 1 {
            might_panic(true);
        } else if countdown < 5 {
            might_panic(false);
        }
        countdown -= 1;
    }
    Ok(())
}

// Notes:
//   1. Compare this program and its coverage results to those of the similar tests `abort.rs` and
//      `try_error_result.rs`.
//   2. Since the `panic_unwind.rs` test is allowed to unwind, it is also allowed to execute the
//      normal program exit cleanup, including writing out the current values of the coverage
//      counters.
//   3. The coverage results show (interestingly) that the `panic!()` call did execute, but it does
//      not show coverage of the `if countdown == 1` branch in `main()` that calls
//      `might_panic(true)` (causing the call to `panic!()`).
//   4. The reason `main()`s `if countdown == 1` branch, calling `might_panic(true)`, appears
//      "uncovered" is, InstrumentCoverage (intentionally) treats `TerminatorKind::Call` terminators
//      as non-branching, because when a program executes normally, they always are. Errors handled
//      via the try `?` operator produce error handling branches that *are* treated as branches in
//      coverage results. By treating calls without try `?` operators as non-branching (assumed to
//      return normally and continue) the coverage graph can be simplified, producing smaller,
//      faster binaries, and cleaner coverage results.
//   5. The reason the coverage results actually show `panic!()` was called is most likely because
//      `panic!()` is a macro, not a simple function call, and there are other `Statement`s and/or
//      `Terminator`s that execute with a coverage counter before the panic and unwind occur.
//   6. Since the common practice is not to use `panic!()` for error handling, the coverage
//      implementation avoids incurring an additional cost (in program size and execution time) to
//      improve coverage results for an event that is generally not "supposed" to happen.
//   7. FIXME(#78544): This issue describes a feature request for a proposed option to enable
//      more accurate coverage results for tests that intentionally panic.
