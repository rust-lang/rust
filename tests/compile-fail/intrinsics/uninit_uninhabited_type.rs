// error-pattern: the evaluated program aborted execution: attempted to instantiate uninhabited type `!`
#![feature(never_type)]

#[allow(deprecated, invalid_value)]
fn main() {
    unsafe { std::mem::uninitialized::<!>() };
}
