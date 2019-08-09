// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// pretty-expanded FIXME #23616

pub trait Writer {
    fn write(&mut self, b: &[u8]) -> Result<(), ()>;
}

fn foo(a: &mut dyn Writer) {
    a.write(&[]).unwrap();
}

pub fn main(){}
