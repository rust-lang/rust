//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Exercises change in <https://github.com/rust-lang/rust/pull/138176>.

struct MyType<'a, T: ?Sized>(&'a (), T);

fn is_sized<T>() {}

trait Id {
    type This: ?Sized;
}
impl<T: ?Sized> Id for T {
    type This = T;
}

fn foo<'a, T: ?Sized>()
where
    (MyType<'a, <T as Id>::This>,): Sized,
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
