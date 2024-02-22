#![allow(unused_assignments)]
//@ failure-status: 101

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
