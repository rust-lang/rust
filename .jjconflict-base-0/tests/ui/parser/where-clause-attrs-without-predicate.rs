// Regression test for <https://github.com/rust-lang/rust/issues/155073>

#![crate_type = "lib"]
#![feature(where_clause_attrs)]

fn f<T>()
where
    T: Copy,
    #[cfg(true)]
    #[cfg(false)]
    //~^ ERROR attribute without where predicates
{
}

fn g<T>()
where
    T: Copy,
    /// dangling
    //~^ ERROR found a documentation comment that doesn't document anything
{
}
