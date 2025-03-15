//@ run-fail
//@ check-run-results
//@ compile-flags: -Zlocation-detail=line,column,file,cstr
//@ exec-env:RUST_BACKTRACE=0

fn main() {
    panic!("with-nul-terminator");
}
