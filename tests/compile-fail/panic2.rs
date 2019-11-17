// ignore-test: Abort panics are not yet supported
// error-pattern: the evaluated program panicked
// compile-flags: -C panic=abort

fn main() {
    std::panic!("{}-panicking from libstd", 42);
}
