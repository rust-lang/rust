// error-pattern: the evaluated program aborted
#![feature(unwind_attributes)]

#[unwind(aborts)]
fn panic_abort() { panic!() }

fn main() {
    panic_abort();
}
