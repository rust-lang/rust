// Check that a `ManuallyDrop` constant holding a valid reference to a nested
// aggregate can still be used as a pattern. The dangling version of this
// constant is rejected in `manually-drop-dangling-ref.rs`.
//
//@ check-pass
use std::mem::ManuallyDrop;

#[derive(PartialEq)]
struct W<T, U>(T, U);

const DATA: ManuallyDrop<&W<i32, W<i32, i32>>> = ManuallyDrop::new(&W(1, W(2, 3)));

fn main() {
    match ManuallyDrop::new(&W(1, W(2, 3))) {
        DATA => {}
        _ => {}
    }
}
