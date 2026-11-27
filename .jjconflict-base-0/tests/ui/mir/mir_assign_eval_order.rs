// Test evaluation order of assignment expressions is right to left.

//@ run-pass

// We would previously not finish evaluating borrow and FRU expressions before
// starting on the LHS

struct S(i32);

fn evaluate_reborrow_before_assign() {
    let mut x = &1;
    let y = &mut &2;
    let z = &3;
    // There's an implicit reborrow of `x` on the right-hand side of the
    // assignment. Note that writing an explicit reborrow would not show this
    // bug, as now there would be two reborrows on the right-hand side and at
    // least one of them would happen before the left-hand side is evaluated.
    *{ x = z; &mut *y } = x;
    assert_eq!(*x, 3);
    assert_eq!(**y, 1);             // y should be assigned the original value of `x`.
}

fn evaluate_mut_reborrow_before_assign() {
    let mut x = &mut 1;
    let y = &mut &mut 2;
    let z = &mut 3;
    *{ x = z; &mut *y } = x;
    assert_eq!(*x, 3);
    assert_eq!(**y, 1);            // y should be assigned the original value of `x`.
}

// We should evaluate `x[2]` and borrow the value out *before* evaluating the
// LHS and changing its value.
fn evaluate_ref_to_temp_before_assign_slice() {
    let mut x = &[S(0), S(1), S(2)][..];
    let y = &mut &S(7);
    *{ x = &[S(3), S(4), S(5)]; &mut *y } = &x[2];
    assert_eq!(2, y.0);
    assert_eq!(5, x[2].0);
}

// We should evaluate `x[2]` and copy the value out *before* evaluating the LHS
// and changing its value.
fn evaluate_fru_to_temp_before_assign_slice() {
    let mut x = &[S(0), S(1), S(2)][..];
    let y = &mut S(7);
    *{ x = &[S(3), S(4), S(5)]; &mut *y } = S { ..x[2] };
    assert_eq!(2, y.0);
    assert_eq!(5, x[2].0);
}

// We should evaluate `*x` and copy the value out *before* evaluating the LHS
// and dropping `x`.
fn evaluate_fru_to_temp_before_assign_box() {
    let x = Box::new(S(0));
    let y = &mut S(1);
    *{ drop(x); &mut *y } = S { ..*x };
    assert_eq!(0, y.0);
}

fn main() {
    evaluate_reborrow_before_assign();
    evaluate_mut_reborrow_before_assign();
    evaluate_ref_to_temp_before_assign_slice();
    evaluate_fru_to_temp_before_assign_slice();
    evaluate_fru_to_temp_before_assign_box();
}
