//@ run-fail
//@ check-run-results:explicit panic
//@ ignore-emscripten no processes

pub fn main() {
    panic!();
    println!("{}", 1);
}
