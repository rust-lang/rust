// run-fail
// error-pattern:attempt to calculate the remainder with a divisor of zero
// ignore-emscripten no processes

#[allow(unconditional_panic)]
fn main() {
    let y = 0;
    let _z = 1 % y;
}
