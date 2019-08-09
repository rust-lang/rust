// Regression test for issue 57464.
//
// Closure are (surprisingly) allowed to outlive their signature. As such it
// was possible to end up with `ReScope`s appearing in the concrete type of an
// opaque type. As all regions are now required to outlive the bound in an
// opaque type we avoid the issue here.

// build-pass (FIXME(62277): could be check-pass?)

struct A<F>(F);

unsafe impl <'a, 'b, F: Fn(&'a i32) -> &'b i32> Send for A<F> {}

fn wrapped_closure() -> impl Sized {
    let f = |x| x;
    f(&0);
    A(f)
}

fn wrapped_closure_with_bound() -> impl Sized + 'static {
    let f = |x| x;
    f(&0);
    A(f)
}

fn main() {
    let x: Box<dyn Send> = Box::new(wrapped_closure());
    let y: Box<dyn Send> = Box::new(wrapped_closure_with_bound());
}
