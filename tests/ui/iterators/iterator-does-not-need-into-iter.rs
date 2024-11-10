//! regression test for #127511: don't suggest `.into_iter()` on iterators

trait Missing {}
trait HasMethod {
    fn foo(self);
}
impl<T: Iterator + Missing> HasMethod for T {
    fn foo(self) {}
}

fn get_iter() -> impl Iterator {
    core::iter::once(())
}

fn main() {
    get_iter().foo();
    //~^ ERROR the method `foo` exists for opaque type `impl Iterator`, but its trait bounds were not satisfied [E0599]
}
