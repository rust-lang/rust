// pretty-expanded FIXME #23616

#![feature(box_syntax)]

fn foo(x: &mut Box<u8>) {
    *x = box 5;
}

pub fn main() {
    foo(&mut box 4);
}
