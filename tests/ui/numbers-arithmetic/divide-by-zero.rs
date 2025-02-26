//@ run-fail
//@ error-pattern:attempt to divide by zero
//@ needs-subprocess

#[allow(unconditional_panic)]
fn main() {
    let y = 0;
    let _z = 1 / y;
}
