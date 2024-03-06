//@ run-fail
//@ error-pattern:attempt to divide by zero
//@ ignore-emscripten no processes

#[allow(unconditional_panic)]
fn main() {
    let y = 0;
    let _z = 1 / y;
}
