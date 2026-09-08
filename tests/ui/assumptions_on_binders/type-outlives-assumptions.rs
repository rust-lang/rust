//@ check-pass
//@ compile-flags: -Zassumptions-on-binders

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
