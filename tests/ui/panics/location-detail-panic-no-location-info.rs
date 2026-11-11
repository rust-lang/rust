//@ run-fail
//@ check-run-results
//@ compile-flags: -Zlocation-detail=none
//@ exec-env:RUST_BACKTRACE=0

fn main() {
    panic!("no location info");
}
