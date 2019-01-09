// run-pass
#![feature(box_syntax)]

struct J { j: isize }

pub fn main() {
    let i: Box<_> = box J {
        j: 100
    };
    assert_eq!(i.j, 100);
}
