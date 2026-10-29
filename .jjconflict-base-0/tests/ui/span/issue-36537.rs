fn main() {
    let p;
    {
        let a = 42;
        p = &a;
        //~^ ERROR `a` does not live long enough

    }
    p.use_ref();

}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
