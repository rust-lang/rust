// See issue #100696.
//@ run-fail
//@ check-run-results
//@ exec-env:RUST_BACKTRACE=0
fn main() {
    &""[1..];
}
