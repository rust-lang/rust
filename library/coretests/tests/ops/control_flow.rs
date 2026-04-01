use core::intrinsics::discriminant_value;
use core::ops::ControlFlow;

#[test]
fn control_flow_discriminants_match_result() {
    // This isn't stable surface area, but helps keep `?` cheap between them,
    // even if LLVM can't always take advantage of it right now.
    // (Sadly Result and Option are inconsistent, so ControlFlow can't match both.)

    assert_eq!(
        discriminant_value(&ControlFlow::<i32, i32>::Break(3)),
        discriminant_value(&Result::<i32, i32>::Err(3)),
    );
    assert_eq!(
        discriminant_value(&ControlFlow::<i32, i32>::Continue(3)),
        discriminant_value(&Result::<i32, i32>::Ok(3)),
    );
}

#[test]
fn control_flow_break_ok() {
    assert_eq!(ControlFlow::<char, i32>::Break('b').break_ok(), Ok('b'));
    assert_eq!(ControlFlow::<char, i32>::Continue(3).break_ok(), Err(3));
}

#[test]
fn control_flow_continue_ok() {
    assert_eq!(ControlFlow::<char, i32>::Break('b').continue_ok(), Err('b'));
    assert_eq!(ControlFlow::<char, i32>::Continue(3).continue_ok(), Ok(3));
}
