// run-fail
//@error-in-other-file:attempt to divide by zero
//@ignore-target-emscripten no processes

#[allow(unconditional_panic)]
fn main() {
    let y = 0;
    let _z = 1 / y;
}
