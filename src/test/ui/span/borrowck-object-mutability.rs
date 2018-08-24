trait Foo {
    fn borrowed(&self);
    fn borrowed_mut(&mut self);
}

fn borrowed_receiver(x: &Foo) {
    x.borrowed();
    x.borrowed_mut(); //~ ERROR cannot borrow
}

fn borrowed_mut_receiver(x: &mut Foo) {
    x.borrowed();
    x.borrowed_mut();
}

fn owned_receiver(x: Box<Foo>) {
    x.borrowed();
    x.borrowed_mut(); //~ ERROR cannot borrow
}

fn mut_owned_receiver(mut x: Box<Foo>) {
    x.borrowed();
    x.borrowed_mut();
}

fn main() {}
