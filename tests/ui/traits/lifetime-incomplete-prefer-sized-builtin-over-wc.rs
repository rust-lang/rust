struct MyType<'a, T: ?Sized>(&'a (), T);

fn is_sized<T>() {}

fn foo<'a, T: ?Sized>()
where
    (MyType<'a, T>,): Sized,
    //~^ ERROR mismatched types
    MyType<'static, T>: Sized,
{
    // Preferring the builtin `Sized` impl of tuples
    // requires proving `MyType<'a, T>: Sized` which
    // can only be proven by using the where-clause,
    // adding an unnecessary `'static` constraint.
    is_sized::<(MyType<'a, T>,)>();
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
