//@ check-pass
//@ compile-flags: -Zassumptions-on-binders

// Regression test for rust-lang/project-assumptions-on-binders#19, minimized from `syn`.
//
// Proving `I: '_` for the `&'_ self` receiver destructures to an OR over every region which
// `I` is known to outlive. Two things are needed for that OR to be satisfiable:
//
// - the implied bound `I: '_` from `&'_ self` has to be part of the root assumptions, not just
//   the explicit `I: 'a` where clause, as `'a: '_` does not hold
// - the resulting `RegionOutlives('_, '_)` candidate has to be discharged, which happens when
//   evaluating the constraint rather than when leaving a universe

trait IterTrait<'a, T: 'a>: Iterator<Item = &'a T> {
    fn clone_box(&self) -> Box<dyn IterTrait<'a, T> + 'a>;
}

impl<'a, T, I> IterTrait<'a, T> for I
where
    T: 'a,
    I: Iterator<Item = &'a T> + Clone + 'a,
{
    fn clone_box(&self) -> Box<dyn IterTrait<'a, T> + 'a> {
        Box::new(self.clone())
    }
}

fn main() {}
