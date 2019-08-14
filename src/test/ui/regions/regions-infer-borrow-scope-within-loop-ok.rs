// run-pass
#![feature(box_syntax)]

fn borrow<T>(x: &T) -> &T {x}

pub fn main() {
    let x: Box<_> = box 3;
    loop {
        let y = borrow(&*x);
        assert_eq!(*x, *y);
        break;
    }
}
