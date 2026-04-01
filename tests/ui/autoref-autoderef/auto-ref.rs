//@ run-pass
struct Foo {
    x: isize,
}

trait Stuff {
    fn printme(&self);
}

impl Stuff for Foo {
    fn printme(&self) {
        println!("{}", self.x);
    }
}

pub fn main() {
    let x = Foo { x: 3 };
    x.printme();
}
