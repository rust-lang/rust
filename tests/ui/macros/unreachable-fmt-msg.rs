// run-fail
//@error-in-other-file:internal error: entered unreachable code: 6 is not prime
//@ignore-target-emscripten no processes

fn main() {
    unreachable!("{} is not {}", 6u32, "prime");
}
