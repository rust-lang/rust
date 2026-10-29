// Test range syntax - borrow errors.

pub fn main() {
    let r = {
        let a = 42;
        let b = 42;
        &a..&b
    };
    //~^^ ERROR `a` does not live long enough
    //~| ERROR `b` does not live long enough
    r.use_ref();
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
