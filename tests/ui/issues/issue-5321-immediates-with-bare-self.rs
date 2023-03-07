// run-pass

trait Fooable {
    fn yes(self);
}

impl Fooable for usize {
    fn yes(self) {
        for _ in 0..self { println!("yes"); }
    }
}

pub fn main() {
    2.yes();
}
