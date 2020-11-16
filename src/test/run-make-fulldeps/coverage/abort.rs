#![feature(unwind_attributes)]
#![allow(unused_assignments)]

#[unwind(aborts)]
fn might_abort(should_abort: bool) {
    if should_abort {
        println!("aborting...");
        panic!("panics and aborts");
    } else {
        println!("Don't Panic");
    }
}

fn main() -> Result<(),u8> {
    let mut countdown = 10;
    while countdown > 0 {
        if countdown < 5 {
            might_abort(false);
        }
        countdown -= 1;
    }
    Ok(())
}

// Notes:
//   1. Compare this program and its coverage results to those of the similar tests
//      `panic_unwind.rs` and `try_error_result.rs`.
//   2. This test confirms the coverage generated when a program includes `TerminatorKind::Abort`.
//   3. The test does not invoke the abort. By executing to a successful completion, the coverage
//      results show where the program did and did not execute.
//   4. If the program actually aborted, the coverage counters would not be saved (which "works as
//      intended"). Coverage results would show no executed coverage regions.
//   6. If `should_abort` is `true` and the program aborts, the program exits with a `132` status
//      (on Linux at least).
