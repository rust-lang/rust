//@error-pattern: memory leaked
//@normalize-stderr-test: ".*│.*" -> "$$stripped$$"

fn main() {
    std::mem::forget(Box::new(42));
}
