// run-fail
// error-pattern:panicked at 'Box<Any>'
// ignore-emscripten no processes

fn main() {
    panic!(Box::new(612_i64));
}
