// error-pattern: the program aborted execution
// compile-flags: -C panic=abort

fn main() {
    core::panic!("panicking from libcore");
}
