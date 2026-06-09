fn f() {
    let mut x = vec![1].iter();
    //~^ ERROR temporary value dropped while borrowed
    x.use_mut();
}

fn main() {
    f();
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
