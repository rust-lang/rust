//@ run-crash
//@ exec-env:RUST_BACKTRACE=0
//@ check-run-results
//@ error-pattern: panic in a destructor during cleanup
//@ normalize-stderr: "\n +[0-9]+:[^\n]+" -> ""
//@ normalize-stderr: "\n +at [^\n]+" -> ""
//@ normalize-stderr: "(core/src/panicking\.rs):[0-9]+:[0-9]+" -> "$1:$$LINE:$$COL"
//@ needs-unwind
//@ ignore-emscripten "RuntimeError" junk in output
//@ ignore-msvc SEH doesn't do panic-during-cleanup the same way as everyone else

struct Bomb;

impl Drop for Bomb {
    fn drop(&mut self) {
        panic!("BOOM");
    }
}

fn main() {
    let _b = Bomb;
    panic!();
}
