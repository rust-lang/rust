// error-pattern: the evaluated program aborted execution
// compile-flags: -C panic=abort

fn main() {
    core::panic!("panicking from libcore");
}
