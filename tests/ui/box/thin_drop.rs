#![feature(thin_box)]
//@ run-pass
use std::boxed::ThinBox;
use std::error::Error;
use std::ops::Deref;
use std::fmt;

fn main() {
    let expected = "Foo error!";
    let mut dropped = false;
    {
        let foo = Foo(expected, &mut dropped);
        let a: ThinBox<dyn Error> = ThinBox::new_unsize(foo);
        let a = a.deref();
        let msg = a.to_string();
        assert_eq!(expected, msg);
    }
    assert!(dropped);
}

#[derive(Debug)]
#[repr(align(1024))]
struct Foo<'a>(&'static str, &'a mut bool);

impl Drop for Foo<'_> {
    fn drop(&mut self) {
        *self.1 = true;
    }
}

impl fmt::Display for Foo<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for Foo<'_> {}
