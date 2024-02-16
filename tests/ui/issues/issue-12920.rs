//@ run-fail
//@ error-pattern:explicit panic
//@ ignore-emscripten no processes

pub fn main() {
    panic!();
    println!("{}", 1);
}
