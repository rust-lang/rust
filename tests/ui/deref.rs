// run-pass
// pretty-expanded FIXME #23616

pub fn main() {
    let x: Box<isize> = Box::new(10);
    let _y: isize = *x;
}
