//! Regression test for <https://github.com/rust-lang/rust/issues/134410> and
//! <https://github.com/rust-lang/rust/issues/111365>.
//!
//! The expected/found notes of "one type is more general than the other" errors
//! used to leak lifetimes bound by outer binders into nested `for<...>` lists,
//! printing invalid types such as `&mut for<'a> fn(for<'a> fn(&'a ()))` or
//! `for<'o> fn(for<'a, 'o> fn(&'a (), &'o ()))`.

type F1 = fn(fn(&'static ()));
type F2 = for<'a> fn(fn(&'a ()));

fn issue_134410(a: &mut F1) {
    let _: &mut F2 = a; //~ ERROR mismatched types
}

type One = fn(HelperOne);
type HelperOne = for<'a> fn(&'a (), &'a ());

type Two = for<'o> fn(HelperTwo<'o>);
type HelperTwo<'x> = for<'a> fn(&'a (), &'x ());

fn issue_111365(x: One) -> Two {
    x //~ ERROR mismatched types
}

fn main() {}
