// run-fail
// check-run-results
// compile-flags: -Zlocation-detail=line,file

fn main() {
    panic!("column-redacted");
}
