// error-pattern: the evaluated program aborted execution
// compile-flags: -C panic=abort
// ignore-windows: windows panics via inline assembly (FIXME)

fn main() {
    core::panic!("{}-panicking from libcore", 42);
}
