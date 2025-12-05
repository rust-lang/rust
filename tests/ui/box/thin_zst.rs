#![feature(thin_box)]
//@ run-pass
use std::boxed::ThinBox;
use std::error::Error;
use std::{fmt, mem};
use std::ops::DerefMut;

const EXPECTED: &str = "boooo!";

fn main() {
    let thin_error: ThinBox<dyn Error> = ThinBox::new_unsize(Foo);
    assert_eq!(mem::size_of::<*const i32>(), mem::size_of_val(&thin_error));
    let msg = thin_error.to_string();
    assert_eq!(EXPECTED, msg);

    let mut thin_concrete_error: ThinBox<Foo> = ThinBox::new(Foo);
    assert_eq!(mem::size_of::<*const i32>(), mem::size_of_val(&thin_concrete_error));
    let msg = thin_concrete_error.to_string();
    assert_eq!(EXPECTED, msg);
    let inner = thin_concrete_error.deref_mut();
    let msg = inner.to_string();
    assert_eq!(EXPECTED, msg);
}

#[derive(Debug)]
struct Foo;

impl fmt::Display for Foo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", EXPECTED)
    }
}

impl Error for Foo {}
