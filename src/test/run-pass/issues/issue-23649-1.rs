// run-pass
use std::mem;

pub struct X([u8]);

fn _f(x: &X) -> usize { match *x { X(ref x) =>  { x.len() } } }

fn main() {
    let b: &[u8] = &[11; 42];
    let v: &X = unsafe { mem::transmute(b) };
    assert_eq!(_f(v), 42);
}
