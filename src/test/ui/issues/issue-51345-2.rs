// run-fail
// error-pattern: thread 'main' panicked at 'explicit panic'
// ignore-emscripten no processes

fn main() {
    let mut vec = vec![];
    vec.push((vec.len(), panic!()));
}
