//@compile-flags: -Zmiri-disable-leak-backtraces
//@error-pattern: the evaluated program leaked memory
//@normalize-stderr-test: ".*│.*" -> "$$stripped$$"

fn main() {
    std::mem::forget(Box::new(42));
}
