// run-pass
// pretty-expanded FIXME #23616

#![feature(box_syntax)]


fn a_val(x: Box<isize>, y: Box<isize>) -> isize {
    *x + *y
}

pub fn main() {
    let z: Box<_> = box 22;
    a_val(z.clone(), z.clone());
}
