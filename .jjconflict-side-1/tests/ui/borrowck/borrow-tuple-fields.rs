struct Foo(Box<isize>, isize);

struct Bar(isize, isize);





fn main() {
    let x: (Box<_>, _) = (Box::new(1), 2);
    let r = &x.0;
    let y = x; //~ ERROR cannot move out of `x` because it is borrowed

    r.use_ref();

    let mut x = (1, 2);
    let a = &x.0;
    let b = &mut x.0; //~ ERROR cannot borrow `x.0` as mutable because it is also borrowed as
    a.use_ref();

    let mut x = (1, 2);
    let a = &mut x.0;
    let b = &mut x.0; //~ ERROR cannot borrow `x.0` as mutable more than once at a time
    a.use_ref();

    let x = Foo(Box::new(1), 2);
    let r = &x.0;
    let y = x; //~ ERROR cannot move out of `x` because it is borrowed
    r.use_ref();

    let mut x = Bar(1, 2);
    let a = &x.0;
    let b = &mut x.0; //~ ERROR cannot borrow `x.0` as mutable because it is also borrowed as
    a.use_ref();

    let mut x = Bar(1, 2);
    let a = &mut x.0;
    let b = &mut x.0; //~ ERROR cannot borrow `x.0` as mutable more than once at a time
    a.use_mut();
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
