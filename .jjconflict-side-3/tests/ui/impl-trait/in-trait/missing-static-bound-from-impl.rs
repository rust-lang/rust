trait Original {
    fn f() -> impl Fn();
}

trait Erased {
    fn f(&self) -> Box<dyn Fn()>;
}

impl<T: Original> Erased for T {
    fn f(&self) -> Box<dyn Fn()> {
        Box::new(<T as Original>::f())
        //~^ ERROR the associated type `impl Fn()` may not live long enough
    }
}

fn main () {}
