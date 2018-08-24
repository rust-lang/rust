// pretty-expanded FIXME #23616

#![feature(box_syntax)]

enum t { foo(Box<isize>), }

pub fn main() {
    let tt = t::foo(box 10);
    match tt { t::foo(_z) => { } }
}
