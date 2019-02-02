// Test that we have enough false edges to avoid exposing the exact matching
// algorithm in borrow checking.

#![feature(nll, bind_by_move_pattern_guards)]

fn guard_always_precedes_arm(y: i32) {
    let mut x;
    // x should always be initialized, as the only way to reach the arm is
    // through the guard.
    match y {
        0 | 2 if { x = 2; true } => x,
        _ => 2,
    };
}

fn guard_may_be_skipped(y: i32) {
    let x;
    // Even though x *is* always initialized, we don't want to have borrowck
    // results be based on whether patterns are exhaustive.
    match y {
        _ if { x = 2; true } => 1,
        _ if {
            x; //~ ERROR use of possibly uninitialized variable: `x`
            false
        } => 2,
        _ => 3,
    };
}

fn guard_may_be_taken(y: bool) {
    let x = String::new();
    // Even though x *is* never moved before the use, we don't want to have
    // borrowck results be based on whether patterns are disjoint.
    match y {
        false if { drop(x); true } => 1,
        true => {
            x; //~ ERROR use of moved value: `x`
            2
        }
        false => 3,
    };
}

fn all_previous_tests_may_be_done(y: &mut (bool, bool)) {
    let r = &mut y.1;
    // We don't actually test y.1 to select the second arm, but we don't want
    // borrowck results to be based on the order we match patterns.
    match y {
        (false, true) => 1, //~ ERROR cannot use `y.1` because it was mutably borrowed
        (true, _) => {
            r;
            2
        }
        (false, _) => 3,
    };
}

fn main() {}
