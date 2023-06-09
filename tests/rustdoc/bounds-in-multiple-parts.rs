#![crate_name = "foo"]

pub trait Eq {}
pub trait Eq2 {}

// Checking that "where predicates" and "generics params" are merged.
// @has 'foo/trait.T.html'
// @has - "//*[@id='tymethod.f']/h4" "fn f<'a, 'b, 'c, T>()where Self: Eq, T: Eq + 'a, 'c: 'b + 'a,"
pub trait T {
    fn f<'a, 'b, 'c: 'a, T: Eq + 'a>()
        where Self: Eq, Self: Eq, T: Eq, 'c: 'b;
}

// Checking that a duplicated "where predicate" is removed.
// @has 'foo/trait.T2.html'
// @has - "//*[@id='tymethod.f']/h4" "fn f<T>()where Self: Eq + Eq2, T: Eq2 + Eq,"
pub trait T2 {
    fn f<T: Eq>()
        where Self: Eq, Self: Eq2, T: Eq2;
}
