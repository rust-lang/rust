//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn main() {
    unreachable!("{} is not {}", 6u32, "prime");
}
