//! Regression test for <https://github.com/rust-lang/rust/issues/62201>.
//!
//! A closure whose parameter type is a higher-ranked associated-type projection
//! (`for<'r> Fn(<Self as Ty<'r>>::V)`) used to be wrongly rejected. It should compile.

//@ check-pass

#![allow(unused, unreachable_code)]

trait Ty<'a> {
    type V;
}

trait SIter: for<'a> Ty<'a> {
    fn f<F>(&self, f: F)
    where
        F: for<'r> Fn(<Self as Ty<'r>>::V);
}

struct S<I>(I);

impl<'a, I: Ty<'a>> Ty<'a> for S<I> {
    type V = <I as Ty<'a>>::V;
}

impl<I: SIter, Item> SIter for S<I>
where
    for<'r> S<I>: Ty<'r, V = Item>,
    for<'r> I: Ty<'r, V = Item>,
{
    fn f<F>(&self, f: F)
    where
        F: Fn(<Self as Ty>::V),
    {
        self.0.f(|item| {
            let item: <Self as Ty>::V = loop {};
            f(item)
        })
    }
}

fn main() {}
