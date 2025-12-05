//@ run-crash
//@ exec-env:RUST_BACKTRACE=0
//@ check-run-results
//@ error-pattern: panic in a function that cannot unwind
//@ error-pattern: Noisy Drop
//@ normalize-stderr: "\n +[0-9]+:[^\n]+" -> ""
//@ normalize-stderr: "\n +at [^\n]+" -> ""
//@ normalize-stderr: "(core/src/panicking\.rs):[0-9]+:[0-9]+" -> "$1:$$LINE:$$COL"
//@ needs-unwind
//@ ignore-emscripten "RuntimeError" junk in output

struct Noise;
impl Drop for Noise {
    fn drop(&mut self) {
        eprintln!("Noisy Drop");
    }
}

extern "C" fn panic_in_ffi() {
    let _val = Noise;
    panic!("Test");
}

fn main() {
    panic_in_ffi();
}
