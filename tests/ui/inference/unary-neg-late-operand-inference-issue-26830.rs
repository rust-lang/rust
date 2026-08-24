//! Regression test for https://github.com/rust-lang/rust/issues/26830.
//! Unary negation should allow later constraints to determine its operand type.

//@ check-pass

fn make<T>() -> T {
    loop {}
}

fn constrained_by_value() {
    let input = make();
    let output = -input;
    let _: i32 = input;
    let _: i32 = output;
}

fn constrained_by_borrow() {
    let input = make();
    let output = -input;
    let _: &i32 = &input;
    let _: i32 = output;
}

fn main() {
    constrained_by_value();
    constrained_by_borrow();
}
