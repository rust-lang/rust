// Test method calls with self as an argument cannot subvert borrow checking.



struct Foo;

impl Foo {
    fn bar(&self) {}
    fn baz(&mut self) {}
}

fn main() {
    let mut x = Foo;
    let y = &mut x;
    Foo::bar(&x); //~ERROR cannot borrow `x`
    y.use_mut();

    let mut x = Foo;
    let y = &mut x;
    Foo::baz(&mut x); //~ERROR cannot borrow `x`
    y.use_mut();
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
