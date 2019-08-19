// Test that we have enough false edges to avoid exposing the exact matching
// algorithm in borrow checking.

#![feature(bind_by_move_pattern_guards)]

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

fn main() {}
