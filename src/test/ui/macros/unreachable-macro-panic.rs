// run-fail
// error-pattern:internal error: entered unreachable code
// ignore-emscripten no processes

fn main() {
    unreachable!()
}
