#![feature(thin_box)]
//@ run-pass
use std::boxed::ThinBox;
use std::error::Error;
use std::{fmt, mem};

fn main() {
    let thin_error: ThinBox<dyn Error> = ThinBox::new_unsize(Foo);
    assert_eq!(mem::size_of::<*const i32>(), mem::size_of_val(&thin_error));
    println!("{:?}", thin_error);

    let thin = ThinBox::new(42i32);
    assert_eq!(mem::size_of::<*const i32>(), mem::size_of_val(&thin));
    println!("{:?}", thin);

    let thin_slice = ThinBox::<[i32]>::new_unsize([1, 2, 3, 4]);
    assert_eq!(mem::size_of::<*const i32>(), mem::size_of_val(&thin_slice));
    println!("{:?}", thin_slice);
}

#[derive(Debug)]
struct Foo;

impl fmt::Display for Foo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "boooo!")
    }
}

impl Error for Foo {}
