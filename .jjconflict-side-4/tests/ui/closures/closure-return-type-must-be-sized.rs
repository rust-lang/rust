#![feature(unboxed_closures)]

trait A {
    fn a() where Self: Sized;
}

mod a {
    use crate::A;

    pub fn foo<F: FnOnce<()>>() where F::Output: A {
        F::Output::a()
    }

    pub fn bar<F: FnOnce() -> R, R: ?Sized>() {}

    pub fn baz<F: FnOnce<()>>() where F::Output: A, F::Output: Sized {
        F::Output::a()
    }
}

mod b {
    use crate::A;

    pub fn foo<F: Fn<()>>() where F::Output: A {
        F::Output::a()
    }

    pub fn bar<F: Fn() -> R, R: ?Sized>() {}

    pub fn baz<F: Fn<()>>() where F::Output: A, F::Output: Sized {
        F::Output::a()
    }
}

mod c {
    use crate::A;

    pub fn foo<F: FnMut<()>>() where F::Output: A {
        F::Output::a()
    }

    pub fn bar<F: FnMut() -> R, R: ?Sized>() {}

    pub fn baz<F: FnMut<()>>() where F::Output: A, F::Output: Sized {
        F::Output::a()
    }
}

impl A for Box<dyn A> {
    fn a() {}
}

fn main() {
    a::foo::<fn() -> dyn A>();         //~ ERROR E0277
    a::bar::<fn() -> dyn A, _>();      //~ ERROR E0277
    a::baz::<fn() -> dyn A>();         //~ ERROR E0277
    a::foo::<fn() -> Box<dyn A>>();    //  ok
    a::bar::<fn() -> Box<dyn A>, _>(); //  ok
    a::baz::<fn() -> Box<dyn A>>();    //  ok

    b::foo::<fn() -> dyn A>();         //~ ERROR E0277
    b::bar::<fn() -> dyn A, _>();      //~ ERROR E0277
    b::baz::<fn() -> dyn A>();         //~ ERROR E0277
    b::foo::<fn() -> Box<dyn A>>();    //  ok
    b::bar::<fn() -> Box<dyn A>, _>(); //  ok
    b::baz::<fn() -> Box<dyn A>>();    //  ok

    c::foo::<fn() -> dyn A>();         //~ ERROR E0277
    c::bar::<fn() -> dyn A, _>();      //~ ERROR E0277
    c::baz::<fn() -> dyn A>();         //~ ERROR E0277
    c::foo::<fn() -> Box<dyn A>>();    //  ok
    c::bar::<fn() -> Box<dyn A>, _>(); //  ok
    c::baz::<fn() -> Box<dyn A>>();    //  ok
}
