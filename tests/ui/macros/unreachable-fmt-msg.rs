//@ run-fail
//@ error-pattern:internal error: entered unreachable code: 6 is not prime
//@ needs-subprocess

fn main() {
    unreachable!("{} is not {}", 6u32, "prime");
}
