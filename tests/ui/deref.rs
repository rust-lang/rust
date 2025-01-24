//@ run-pass

pub fn main() {
    let x: Box<isize> = Box::new(10);
    let _y: isize = *x;
}
