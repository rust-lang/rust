// run-fail
// check-run-results
// compile-flags: -Zlocation-detail=line,column
// exec-env:RUST_BACKTRACE=0
// ignore-emscripten has extra panic output

fn main() {
    let opt: Option<u32> = None;
    opt.unwrap();
}
