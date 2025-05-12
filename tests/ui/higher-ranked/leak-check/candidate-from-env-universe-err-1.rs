//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@[old] check-pass

// cc #119820

trait Trait {}

impl<T: Trait> Trait for &T {}
impl Trait for u32 {}

fn hr_bound<T>()
where
    for<'a> &'a T: Trait,
{
}

fn foo<T>()
where
    T: Trait,
    for<'a> &'a &'a T: Trait,
{
    // We get a universe error when using the `param_env` candidate
    // but are able to successfully use the impl candidate. Without
    // the leak check both candidates may apply and we prefer the
    // `param_env` candidate in winnowing.
    hr_bound::<&T>();
    //[next]~^ ERROR the trait bound `for<'a> &'a &T: Trait` is not satisfied
}

fn main() {}
