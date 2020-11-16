#![allow(unused_assignments, unused_variables)]

fn main() {
    let mut countdown = 0;
    if true {
        countdown = 10;
    }

    const B: u32 = 100;
    let x = if countdown > 7 {
        countdown -= 4;
        B
    } else if countdown > 2 {
        if countdown < 1 || countdown > 5 || countdown != 9 {
            countdown = 0;
        }
        countdown -= 5;
        countdown
    } else {
        return;
    };

    let mut countdown = 0;
    if true {
        countdown = 10;
    }

    if countdown > 7 {
        countdown -= 4;
    } else if countdown > 2 {
        if countdown < 1 || countdown > 5 || countdown != 9 {
            countdown = 0;
        }
        countdown -= 5;
    } else {
        return;
    }

    if true {
        // Demonstrate the difference with `TerminatorKind::Assert` as of 2020-11-15. Assert is no
        // longer treated as a `BasicCoverageBlock` terminator, which changed the coverage region,
        // for the executed `then` block above, to include the closing brace on line 30. That
        // changed the line count, but the coverage code region (for the `else if` condition) is
        // still valid.
        //
        // Note that `if` (then) and `else` blocks include the closing brace in their coverage
        // code regions when the last line in the block ends in a semicolon, because the Rust
        // compiler inserts a `StatementKind::Assign` to assign `const ()` to a `Place`, for the
        // empty value for the executed block. When the last line does not end in a semicolon
        // (that is, when the block actually results in a value), the additional `Assign` is not
        // generated, and the brace is not included.
        let mut countdown = 0;
        if true {
            countdown = 10;
        }

        if countdown > 7 {
            countdown -= 4;
        }
        // The closing brace of the `then` branch is now included in the coverage region, and shown
        // as "executed" (giving its line a count of 1 here). Since, in the original version above,
        // the closing brace shares the same line as the `else if` conditional expression (which is
        // not executed if the first `then` condition is true), only the condition's code region is
        // marked with a count of 0 now.
        else if countdown > 2 {
            if countdown < 1 || countdown > 5 || countdown != 9 {
                countdown = 0;
            }
            countdown -= 5;
        } else {
            return;
        }
    }

    let mut countdown = 0;
    if true {
        countdown = 1;
    }

    let z = if countdown > 7 {
        countdown -= 4;
    } else if countdown > 2 {
        if countdown < 1 || countdown > 5 || countdown != 9 {
            countdown = 0;
        }
        countdown -= 5;
    } else {
        let should_be_reachable = countdown;
        println!("reached");
        return;
    };

    let w = if countdown > 7 {
        countdown -= 4;
    } else if countdown > 2 {
        if countdown < 1 || countdown > 5 || countdown != 9 {
            countdown = 0;
        }
        countdown -= 5;
    } else {
        return;
    };
}
