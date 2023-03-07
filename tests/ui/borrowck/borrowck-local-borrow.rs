// run-fail
// error-pattern:panic 1
// ignore-emscripten no processes

fn main() {
    let x = 2;
    let y = &x;
    panic!("panic 1");
}
