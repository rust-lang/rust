// run-pass
// pretty-expanded FIXME #23616

#![allow(path_statements)]

pub fn main() {
    let y: Box<_> = Box::new(1);
    y;
}
