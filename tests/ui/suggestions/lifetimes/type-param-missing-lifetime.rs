// We want to suggest a bound `T: 'a` but `'a` is elided,
//@ run-rustfix
//@ edition: 2018
#![allow(dead_code)]

struct Inv<'a>(Option<*mut &'a u8>);

fn check_bound<'a, A: 'a>(_: A, _: Inv<'a>) {}

struct Elided<'a, T = ()>(Inv<'a>, T);

struct MyTy<X>(X);

impl<X> MyTy<Elided<'_, X>> {
    async fn foo<A>(self, arg: A, _: &str) -> &str {
        check_bound(arg, self.0 .0);
        //~^ ERROR parameter type `A` may not live long enough
        ""
    }
}

// Make sure the new lifetime name doesn't conflict with
// other early- or late-bound lifetimes in-scope.
impl<'a, A> MyTy<(A, &'a ())> {
    async fn foo2(
        arg: A,
        lt: Inv<'_>,
    ) -> (
        impl Into<&str> + Into<&'_ str> + '_,
        impl Into<Option<Elided>> + '_,
        impl Into<Option<Elided<()>>>,
    ) {
        check_bound(arg, lt);
        //~^ ERROR parameter type `A` may not live long enough
        ("", None, None)
    }

    // same as above but there is a late-bound lifetime named `'b`.
    async fn bar2<'b>(_dummy: &'a u8, arg: A, lt: Inv<'_>) {
        check_bound(arg, lt);
        //~^ ERROR parameter type `A` may not live long enough
    }
}

impl<A> MyTy<Elided<'_, A>> {
    async fn foo3(self) {
        check_bound(self.0 .1, self.0 .0);
        //~^ ERROR parameter type `A` may not live long enough
    }
}

fn main() {}
