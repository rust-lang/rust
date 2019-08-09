// run-pass

struct X {
    a: isize
}

trait Changer {
    fn change(self) -> Self;
}

impl Changer for X {
    fn change(mut self) -> X {
        self.a = 55;
        self
    }
}

pub fn main() {
    let x = X { a: 32 };
    let new_x = x.change();
    assert_eq!(new_x.a, 55);
}
