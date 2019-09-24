// run-pass
// pretty-expanded FIXME #23616

#![feature(box_syntax)]

pub fn main() {
    fn f() {
    };
    let _: Box<fn()> = box (f as fn());
}
