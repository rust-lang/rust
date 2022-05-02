// run-fail
// check-run-results
// compile-flags: -Zlocation-detail=line,file
// exec-env:RUST_BACKTRACE=0
// ignore-emscripten has extra panic output

fn main() {
    panic!("column-redacted");
}
