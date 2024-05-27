// issue: rust-lang/rust#125520
#![feature(generic_const_exprs)]
//~^ WARN the feature `generic_const_exprs` is incomplete and may not be safe to use and/or cause compiler crashes

struct Outer<const A: i64, const B: i64>();
impl<const A: usize, const B: usize> Outer<A, B>
//~^ ERROR the constant `A` is not of type `i64`
//~| ERROR the constant `B` is not of type `i64`
//~| ERROR mismatched types
//~| ERROR mismatched types
where
    [(); A + (B * 2)]:,
{
    fn i() -> Self {
    //~^ ERROR the constant `A` is not of type `i64`
    //~| ERROR the constant `B` is not of type `i64`
        Self
        //~^ ERROR mismatched types
        //~| ERROR the constant `A` is not of type `i64`
        //~| ERROR the constant `B` is not of type `i64`
    }
}

fn main() {
    Outer::<1, 1>::o();
    //~^ ERROR no function or associated item named `o` found for struct `Outer` in the current scope
}
