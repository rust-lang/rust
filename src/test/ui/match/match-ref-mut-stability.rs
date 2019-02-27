// Check that `ref mut` variables don't change address between the match guard
// and the arm expression.

// run-pass

#![feature(nll, bind_by_move_pattern_guards)]

// Test that z always point to the same temporary.
fn referent_stability() {
    let p;
    match 0 {
        ref mut z if { p = z as *const _; true } => assert_eq!(p, z as *const _),
        _ => unreachable!(),
    };
}

// Test that z is always effectively the same variable.
fn variable_stability() {
    let p;
    match 0 {
        ref mut z if { p = &z as *const _; true } => assert_eq!(p, &z as *const _),
        _ => unreachable!(),
    };
}

// Test that a borrow of *z can cross from the guard to the arm.
fn persist_borrow() {
    let r;
    match 0 {
        ref mut z if { r = z as &_; true } => assert_eq!(*r, 0),
        _ => unreachable!(),
    }
}

fn main() {
    referent_stability();
    variable_stability();
    persist_borrow();
}
