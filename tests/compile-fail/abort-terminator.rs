// error-pattern: the evaluated program aborted
// ignore-windows (panics dont work on Windows)
#![feature(unwind_attributes)]

#[unwind(aborts)]
fn panic_abort() { panic!() }

fn main() {
    panic_abort();
}
