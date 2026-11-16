//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

// A regression test which failed with the old solver. The old solver
// ended up normalizing `<bar as Parser<'!c>>::Output` to `?t` because
// it did not eagerly prove the nested `F: Fn(&'s str) -> (&'s str, T)`
// where-clause.
//
// This means we don't map the `'!c` back to a bound var after
// inferring `?t`. We only normalize inside of binders with the new
// solver if the resulting term does not reference any inference
// variables which can later be constrained to mention something from
// the binder we're in. This is necessary as these placeholders would
// then leak from normalization.

trait Parser<'s> {
    type Output;
}

impl<'s, F, T> Parser<'s> for F
where
    F: Fn(&'s str) -> (&'s str, T),
{
    type Output = T;
}

fn foo<F1, F2>(f1: F1, f2: F2)
where
    F1: for<'a> Parser<'a>,
    F2: for<'b, 'c> FnOnce(&'b <F1 as Parser<'c>>::Output) -> bool,
{
}

struct Wrapper<'a>(&'a str);

fn main() {
    fn bar<'a>(s: &'a str) -> (&'a str, &'a str) {
        (&s[..1], &s[..])
    }

    fn baz<'a>(s: &'a str) -> (&'a str, Wrapper<'a>) {
        (&s[..1], Wrapper(&s[..]))
    }

    foo(bar, |s| s.len() == 5);
    //[current]~^ ERROR implementation of `FnOnce` is not general enough
    //[current]~| ERROR implementation of `FnOnce` is not general enough
    foo(baz, |s| s.0.len() == 5);
    //[current]~^ ERROR implementation of `FnOnce` is not general enough
    //[current]~| ERROR implementation of `FnOnce` is not general enough
}
