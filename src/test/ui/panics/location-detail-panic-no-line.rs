// run-fail
// check-run-results
// compile-flags: -Zlocation-detail=file,column
// exec-env:RUST_BACKTRACE=0
// ignore-emscripten has extra panic output

fn main() {
    panic!("line-redacted");
}
