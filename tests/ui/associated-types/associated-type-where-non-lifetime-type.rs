//! regression test for #149233

trait Foo {
    type Bar<'a>
    where
        Self: Sized;
    fn test(&self);
}
impl Foo for () {
    type Bar<'a>
        = ()
    where
        for<T> T:;
    //~^ ERROR: only lifetime parameters can be used in this context
    fn test(&self) {}
}

fn main() {}
