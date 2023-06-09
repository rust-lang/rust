// run-pass
// pretty-expanded FIXME #23616

pub fn main() {
    let _x: Box<_> = Box::new(vec![0,0,0,0,0]);
}
