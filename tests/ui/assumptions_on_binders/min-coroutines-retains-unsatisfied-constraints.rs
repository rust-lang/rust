//@ compile-flags: -Zassumptions-on-binders=min_coroutines
//@ normalize-stderr: "\n\n$" -> "\n"

#![feature(test_binder_constraints)]
#![allow(internal_features)]

core::test_binder_constraints! {
    impl<'a, 'b> {
        forall<'w> where 'b: 'w {
            //~^ ERROR higher-ranked lifetime bound could not be satisfied
            'b: 'w,
            'a: 'b,
        } expect {
            'a: 'b,
        }
    }
}

fn main() {}
