//@ check-pass
//@ compile-flags: -Znext-solver

trait Local {}

trait Overlap { fn f(); }
impl<T> Overlap for Option<T> where Self: Clone, { fn f() {} }
impl<T> Overlap for Option<T> where Self: Local, { fn f() {} }

fn test<T>()
where
    Option<T>: Clone + Local,
{
    <Option<T> as Overlap>::f();
}

fn main() {}
