// Test that borrows that occur due to calls to object methods
// properly "claim" the object path.



trait Foo {
    fn borrowed(&self) -> &();
    fn mut_borrowed(&mut self) -> &();
}

fn borrowed_receiver(x: &dyn Foo) {
    let y = x.borrowed();
    let z = x.borrowed();
    z.use_ref();
    y.use_ref();
}

fn mut_borrowed_receiver(x: &mut dyn Foo) {
    let y = x.borrowed();
    let z = x.mut_borrowed(); //~ ERROR cannot borrow
    y.use_ref();
}

fn mut_owned_receiver(mut x: Box<dyn Foo>) {
    let y = x.borrowed();
    let z = &mut x; //~ ERROR cannot borrow
    y.use_ref();
}

fn imm_owned_receiver(mut x: Box<dyn Foo>) {
    let y = x.borrowed();
    let z = &x;
    z.use_ref();
    y.use_ref();
}

fn main() {}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
