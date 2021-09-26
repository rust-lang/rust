// run-pass

struct X {
    a: isize
}

trait Changer {
    fn change(self: Box<Self>) -> Box<Self>;
}

impl Changer for X {
    fn change(mut self: Box<X>) -> Box<X> {
        self.a = 55;
        self
    }
}

pub fn main() {
    let x: Box<_> = Box::new(X { a: 32 });
    let new_x = x.change();
    assert_eq!(new_x.a, 55);
}
