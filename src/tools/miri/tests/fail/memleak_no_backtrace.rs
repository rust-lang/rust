//@compile-flags: -Zmiri-disable-leak-backtraces
//@error-in-other-file: memory leaked
//@normalize-stderr-test: ".*│.*" -> "$$stripped$$"

fn main() {
    std::mem::forget(Box::new(42));
}
