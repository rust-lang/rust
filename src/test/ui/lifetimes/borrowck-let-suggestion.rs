fn f() {
    let mut x = vec![1].iter();
    //~^ ERROR borrowed value does not live long enough
    x.use_mut();
}

fn main() {
    f();
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
