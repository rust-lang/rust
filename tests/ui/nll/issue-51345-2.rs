//@ run-fail
//@ check-run-results:thread 'main' panicked
//@ check-run-results:explicit panic
//@ ignore-emscripten no processes

fn main() {
    let mut vec = vec![];
    vec.push((vec.len(), panic!()));
}
