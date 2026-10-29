fn main() {
    let mut x = "foo";
    let y = &mut x;
    let z = &mut x; //~ ERROR cannot borrow
    z.use_mut();
    y.use_mut();
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
