//@ compile-flags: -Znext-solver
//@ revisions: pass fail
//@[pass] check-pass

struct Wrapper<T, const N: usize>([T; N]);

trait Foo {}
fn needs_foo<F: Foo>() {}

#[cfg(fail)]
impl<T> Foo for [T; 1] {}

#[cfg(pass)]
impl<T, const N: usize> Foo for [T; N] {}

fn test<T, const N: usize>() {
    needs_foo::<[T; N]>();
    //[fail]~^ ERROR the trait bound `[T; N]: Foo` is not satisfied
}

fn main() {}
