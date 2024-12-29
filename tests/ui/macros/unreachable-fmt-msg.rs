//@ run-fail
//@ check-run-results:internal error: entered unreachable code: 6 is not prime
//@ ignore-emscripten no processes

fn main() {
    unreachable!("{} is not {}", 6u32, "prime");
}
