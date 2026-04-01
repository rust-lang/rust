fn main() {
    let mut value = 3;
    let _borrow = &mut value;
    let _sum = value + 1; //~ ERROR E0503
    _borrow.use_mut();
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
