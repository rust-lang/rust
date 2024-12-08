trait Foo {
    fn borrowed(&self);
    fn borrowed_mut(&mut self);
}

fn borrowed_receiver(x: &dyn Foo) {
    x.borrowed();
    x.borrowed_mut(); //~ ERROR cannot borrow
}

fn borrowed_mut_receiver(x: &mut dyn Foo) {
    x.borrowed();
    x.borrowed_mut();
}

fn owned_receiver(x: Box<dyn Foo>) {
    x.borrowed();
    x.borrowed_mut(); //~ ERROR cannot borrow
}

fn mut_owned_receiver(mut x: Box<dyn Foo>) {
    x.borrowed();
    x.borrowed_mut();
}

fn main() {}
