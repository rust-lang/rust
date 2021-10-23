// run-fail
// check-run-results
// compile-flags: -Zlocation-detail=file,column

fn main() {
    panic!("line-redacted");
}
