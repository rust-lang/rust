#![crate_type = "lib"]

// This is part of a collection of regression tests related to the NLL problem case 3 that was
// deferred from the implementation of the NLL RFC, and left to be implemented by polonius. They are
// from open issues, e.g. tagged fixed-by-polonius, to ensure that the polonius alpha analysis does
// handle them, as does the datalog implementation.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [nll] known-bug: #68934
//@ [polonius] check-pass
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] check-pass
//@ [legacy] compile-flags: -Z polonius=legacy

enum Either<A, B> {
    Left(A),
    Right(B),
}

enum Tree<'a, A, B> {
    ALeaf(A),
    BLeaf(B),
    ABranch(&'a mut Tree<'a, A, B>, A),
    BBranch(&'a mut Tree<'a, A, B>, B),
}

impl<'a, A: PartialOrd, B> Tree<'a, A, B> {
    fn deep_fetch(&mut self, value: Either<A, B>) -> Result<&mut Self, (&mut Self, Either<A, B>)> {
        match (self, value) {
            (Tree::ABranch(ref mut a, ref v), Either::Left(vv)) if v > &vv => {
                a.deep_fetch(Either::Left(vv))
            }

            (this, _v) => Err((this, _v)),
        }
    }
}
