// https://github.com/rust-lang/rust/issues/8398
//@ check-pass
#![allow(dead_code)]

pub trait Writer {
    fn write(&mut self, b: &[u8]) -> Result<(), ()>;
}

fn foo(a: &mut dyn Writer) {
    a.write(&[]).unwrap();
}

pub fn main(){}
