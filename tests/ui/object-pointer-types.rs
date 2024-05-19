trait Foo {
    fn borrowed(&self);
    fn borrowed_mut(&mut self);

    fn owned(self: Box<Self>);
}

fn borrowed_receiver(x: &dyn Foo) {
    x.borrowed();
    x.borrowed_mut(); // See [1]
    x.owned(); //~ ERROR no method named `owned` found
}

fn borrowed_mut_receiver(x: &mut dyn Foo) {
    x.borrowed();
    x.borrowed_mut();
    x.owned(); //~ ERROR no method named `owned` found
}

fn owned_receiver(x: Box<dyn Foo>) {
    x.borrowed();
    x.borrowed_mut(); // See [1]
    x.managed();  //~ ERROR no method named `managed` found
    x.owned();
}

fn main() {}

// [1]: These cases are illegal, but the error is not detected
// until borrowck, so see the test borrowck-object-mutability.rs
