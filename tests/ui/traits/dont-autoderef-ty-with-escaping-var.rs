// issue:113951

trait Foo<'x, T> {}

trait RefFoo<T> {
    fn ref_foo(&self);
}

impl<T> RefFoo<T> for T
where
    for<'a> &'a mut Vec<&'a u32>: Foo<'static, T>,
{
    fn ref_foo(&self) {}
}

fn coerce_lifetime2() {
    <i32 as RefFoo<i32>>::ref_foo(unknown);
    //~^ ERROR cannot find value `unknown` in this scope
    //~| ERROR trait `Foo<'_, i32>` is not implemented for `&mut Vec<&u32>`
}

fn main() {}
