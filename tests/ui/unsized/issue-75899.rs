//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ check-pass

trait Trait {}
impl<T> Trait for T {}

trait Noop {
    type Assoc: ?Sized;
}
impl<T: ?Sized> Noop for T {
    type Assoc = T;
}

struct NoopNewtype<T: ?Sized + Noop>(T::Assoc);
fn coerce_newtype<T: Trait>(x: &NoopNewtype<T>) -> &NoopNewtype<dyn Trait + '_> {
    x
}

fn main() {}
