//error-pattern: the evaluated program panicked
// compile-flags: -C panic=abort

fn main() {
    core::panic!("panicking from libcore");
}
