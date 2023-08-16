// run-fail
//@error-in-other-file:explicit panic
//@ignore-target-emscripten no processes

pub fn main() {
    panic!();
    println!("{}", 1);
}
