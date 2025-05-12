//@ run-pass

pub fn main() {
    let _x: Box<_> = Box::new(vec![0,0,0,0,0]);
}
