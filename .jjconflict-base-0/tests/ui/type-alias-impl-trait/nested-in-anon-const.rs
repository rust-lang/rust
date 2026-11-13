// Regression test for issue #119295.

#![feature(type_alias_impl_trait)]

type Bar<T> = T;
type S<const A: usize> = [i32; A];

extern "C" {
    pub fn lint_me(
        x: Bar<
            S<
                { //~ ERROR mismatched types
                    type B<Z> = impl Sized;
                    //~^ ERROR unconstrained opaque type
                },
            >,
        >,
    );
}

fn main() {}
