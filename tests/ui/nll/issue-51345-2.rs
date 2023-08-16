// run-fail
//@error-in-other-file: thread 'main' panicked at 'explicit panic'
//@ignore-target-emscripten no processes

fn main() {
    let mut vec = vec![];
    vec.push((vec.len(), panic!()));
}
