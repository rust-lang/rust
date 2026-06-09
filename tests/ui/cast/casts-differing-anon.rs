use std::fmt;

fn foo() -> Box<impl fmt::Debug+?Sized> {
    let x : Box<[u8]> = Box::new([0]);
    x
}
fn bar() -> Box<impl fmt::Debug+?Sized> {
    let y: Box<dyn fmt::Debug> = Box::new([0]);
    y
}

fn main() {
    let f = foo();
    let b = bar();

    // this is an `*mut [u8]` in practice
    let f_raw : *mut _ = Box::into_raw(f);
    // this is an `*mut fmt::Debug` in practice
    let mut b_raw = Box::into_raw(b);
    // ... and they should not be mixable
    b_raw = f_raw as *mut _; //~ ERROR is invalid
}
