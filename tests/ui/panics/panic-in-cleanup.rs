// run-fail
// check-run-results
// error-pattern: panic in a destructor during cleanup
// normalize-stderr-test: "\n +[0-9]+:[^\n]+" -> ""
// normalize-stderr-test: "\n +at [^\n]+" -> ""
// ignore-emscripten no processes

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
