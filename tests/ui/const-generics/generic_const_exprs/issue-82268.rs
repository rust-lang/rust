//@ build-pass

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

trait Collate<Op> {
    type Pass;
    type Fail;

    fn collate(self) -> (Self::Pass, Self::Fail);
}

impl<Op> Collate<Op> for () {
    type Pass = ();
    type Fail = ();

    fn collate(self) -> ((), ()) {
        ((), ())
    }
}

trait CollateStep<X, Prev> {
    type Pass;
    type Fail;
    fn collate_step(x: X, prev: Prev) -> (Self::Pass, Self::Fail);
}

impl<X, P, F> CollateStep<X, (P, F)> for () {
    type Pass = (X, P);
    type Fail = F;

    fn collate_step(x: X, (p, f): (P, F)) -> ((X, P), F) {
        ((x, p), f)
    }
}

struct CollateOpImpl<const MASK: u32>;
trait CollateOpStep {
    type NextOp;
    type Apply;
}

impl<const MASK: u32> CollateOpStep for CollateOpImpl<MASK>
where
    CollateOpImpl<{ MASK >> 1 }>: Sized,
{
    type NextOp = CollateOpImpl<{ MASK >> 1 }>;
    type Apply = ();
}

impl<H, T, Op: CollateOpStep> Collate<Op> for (H, T)
where
    T: Collate<Op::NextOp>,
    Op::Apply: CollateStep<H, (T::Pass, T::Fail)>,
{
    type Pass = <Op::Apply as CollateStep<H, (T::Pass, T::Fail)>>::Pass;
    type Fail = <Op::Apply as CollateStep<H, (T::Pass, T::Fail)>>::Fail;

    fn collate(self) -> (Self::Pass, Self::Fail) {
        <Op::Apply as CollateStep<H, (T::Pass, T::Fail)>>::collate_step(self.0, self.1.collate())
    }
}

fn collate<X, const MASK: u32>(x: X) -> (X::Pass, X::Fail)
where
    X: Collate<CollateOpImpl<MASK>>,
{
    x.collate()
}

fn main() {
    dbg!(collate::<_, 5>(("Hello", (42, ('!', ())))));
}
