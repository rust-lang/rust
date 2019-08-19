// run-pass
#![feature(box_syntax)]

struct X {
    a: isize
}

trait Changer : Sized {
    fn change(mut self) -> Self {
        self.set_to(55);
        self
    }

    fn change_again(mut self: Box<Self>) -> Box<Self> {
        self.set_to(45);
        self
    }

    fn set_to(&mut self, a: isize);
}

impl Changer for X {
    fn set_to(&mut self, a: isize) {
        self.a = a;
    }
}

pub fn main() {
    let x = X { a: 32 };
    let new_x = x.change();
    assert_eq!(new_x.a, 55);

    let x: Box<_> = box new_x;
    let new_x = x.change_again();
    assert_eq!(new_x.a, 45);
}
