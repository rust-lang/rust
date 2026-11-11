// Test that we have enough false edges to avoid exposing the exact matching
// algorithm in borrow checking.

fn all_previous_tests_may_be_done(y: &mut (bool, bool)) {
    let r = &mut y.1;
    // We don't actually test y.1 to select the second arm, but we don't want
    // borrowck results to be based on the order we match patterns.
    match y {
        //~^ ERROR cannot use `y.1` because it was mutably borrowed
        (false, true) => {}
        // Borrowck must not know we don't test `y.1` when `y.0` is `true`.
        (true, _) => drop(r),
        (false, _) => {}
    };

    // Fine in the other order.
    let r = &mut y.1;
    match y {
        (true, _) => drop(r),
        (false, true) => {}
        (false, _) => {}
    };
}

fn main() {}
