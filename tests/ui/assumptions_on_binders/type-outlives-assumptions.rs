//@ check-pass
//@ compile-flags: -Zassumptions-on-binders

#![feature(test_binder_constraints, non_lifetime_binders)]
#![expect(incomplete_features)]

// A type outlives assumption also implies that each region in the type outlives the RHS. The
// unrelated `'c` makes the derived edge visible in the lifted result. Its bounds then satisfy the
// lifted constraint after the outer binder is left.
core::test_binder_constraints! {
    impl<'b, 'c: 'b + 'static> {
        forall<'a> where &'b u8: 'a {
            'c: 'a
        } expect {
            or {
                'c: 'b,
                'c: 'static,
            }
        }
    }
}

// Regions bound inside the type are ignored, but free regions still contribute outlives edges.
core::test_binder_constraints! {
    impl<'b, 'd: 'b + 'static> {
        forall<'a> where for<'c> fn(&'c (), &'b u8): 'a {
            'd: 'a
        } expect {
            or {
                'd: 'b,
                'd: 'static,
            }
        }
    }
}

// Regression test for rust-lang/project-assumptions-on-binders#19, based on the `syn` failure.
// The receiver gives us an implied `I: 'b` bound and `'b: 'a` lets that satisfy the object
// lifetime. The implied type bound has to be included in the root assumptions for that to work.
trait IterTrait<'a, T: 'a>: Iterator<Item = &'a T> {
    fn clone_box<'b>(&'b self) -> Box<dyn IterTrait<'a, T> + 'a>
    where
        'b: 'a;
}

impl<'a, T, I> IterTrait<'a, T> for I
where
    T: 'a,
    I: Iterator<Item = &'a T> + Clone,
{
    fn clone_box<'b>(&'b self) -> Box<dyn IterTrait<'a, T> + 'a>
    where
        'b: 'a,
    {
        Box::new(self.clone())
    }
}

fn main() {}
