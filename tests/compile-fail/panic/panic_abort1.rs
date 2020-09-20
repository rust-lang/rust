// error-pattern: the evaluated program aborted execution
// compile-flags: -C panic=abort

fn main() {
    std::panic!("panicking from libstd");
}
