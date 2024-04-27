//@compile-flags: -Zmiri-disable-leak-backtraces
//@error-in-other-file: the evaluated program leaked memory
//@normalize-stderr-test: ".*│.*" -> "$$stripped$$"

fn main() {
    std::mem::forget(Box::new(42));
}
