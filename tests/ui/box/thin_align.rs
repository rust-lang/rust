#![feature(thin_box)]
//@ run-pass
use std::boxed::ThinBox;
use std::error::Error;
use std::ops::Deref;
use std::fmt;

fn main() {
    let expected = "Foo error!";
    let a: ThinBox<dyn Error> = ThinBox::new_unsize(Foo(expected));
    let a = a.deref();
    let msg = a.to_string();
    assert_eq!(expected, msg);
}

#[derive(Debug)]
#[repr(align(1024))]
struct Foo(&'static str);

impl fmt::Display for Foo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for Foo {}
