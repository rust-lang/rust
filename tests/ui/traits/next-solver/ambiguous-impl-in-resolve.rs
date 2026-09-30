//@ revisions: old next
//@[next] compile-flags: -Znext-solver

// Impossible where-clauses can cause multiple overlapping impls to apply
// with the same constraints.

trait Local {}

trait Overlap { fn f(); }
impl<T> Overlap for Option<T> where Self: Clone, { fn f() {} }
impl<T> Overlap for Option<T> where Self: Local, { fn f() {} }

fn test<T>()
where
    Option<T>: Clone + Local,
{
    <Option<T> as Overlap>::f();
    //~^ ERROR type annotations needed
}

fn main() {}
