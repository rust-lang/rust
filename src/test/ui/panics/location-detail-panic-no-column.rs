// run-fail
// check-run-results
// compile-flags: -Zlocation-detail=line,file
// exec-env:RUST_BACKTRACE=0

fn main() {
    panic!("column-redacted");
}
