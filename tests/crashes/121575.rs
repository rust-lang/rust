//@ known-bug: #121575
// ignore-tidy-linelength
#![feature(generic_const_exprs)]

use std::array;

trait PrimRec<const N: usize, const O: usize> {
    fn eval(&self, x: [usize; N]) -> [usize; O];
}

struct Zero;

impl<const N: usize> PrimRec<N, 1> for Zero {
    fn eval(&self, _: [usize; N]) -> [usize; 1] {
        [0]
    }
}

struct Const(usize);

impl<const N: usize> PrimRec<N, 1> for Const {
    fn eval(&self, _: [usize; N]) -> [usize; 1] {
        [self.0]
    }
}

struct S;

impl PrimRec<1, 1> for S {
    fn eval(&self, x: [usize; 1]) -> [usize; 1] {
        [x[0] + 1]
    }
}

struct Proj<const I: usize>;

impl<const N: usize, const I: usize> PrimRec<N, 1> for Proj<I> {
    fn eval(&self, x: [usize; N]) -> [usize; 1] {
        [x[I]]
    }
}

struct Merge<const N: usize, const O1: usize, const O2: usize, A: PrimRec<N, O1>, B: PrimRec<N, O2>>(
    A,
    B,
);

fn concat<const M: usize, const N: usize>(a: [usize; M], b: [usize; N]) -> [usize; M + N] {
    array::from_fn(|i| if i < M { a[i] } else { b[i - M] })
}

impl<const N: usize, const O1: usize, const O2: usize, A: PrimRec<N, O1>, B: PrimRec<N, O2>>
    PrimRec<N, { O1 + O2 }> for Merge<N, O1, O2, A, B>
{
    fn eval(&self, x: [usize; N]) -> [usize; O1 + O2] {
        concat(self.0.eval(x), self.1.eval(x))
    }
}

struct Compose<const N: usize, const I: usize, const O: usize, A: PrimRec<N, I>, B: PrimRec<I, O>>(
    A,
    B,
);

impl<const N: usize, const I: usize, const O: usize, A: PrimRec<N, I>, B: PrimRec<I, O>>
    PrimRec<N, O> for Compose<N, I, O, A, B>
{
    fn eval(&self, x: [usize; N]) -> [usize; O] {
        self.1.eval(self.0.eval(x))
    }
}

struct Rec<const N: usize, const O: usize, Base: PrimRec<N, O>, F: PrimRec<{ O + (N + 1) }, O>>(
    Base,
    F,
);

fn tail<const N: usize>(x: [usize; N + 1]) -> [usize; N] {
    array::from_fn(|i| x[i + 1])
}

fn cons<const N: usize>(x: usize, xs: [usize; N]) -> [usize; N + 1] {
    array::from_fn(|i| if i == 0 { x } else { xs[i - 1] })
}

impl<const N: usize, const O: usize, Base: PrimRec<N, O>, F: PrimRec<{ O + (N + 1) }, O>>
    PrimRec<{ N + 1 }, O> for Rec<N, O, Base, F>
{
    fn eval(&self, x: [usize; N + 1]) -> [usize; O] {
        match (x[0], tail(x)) {
            (0, x) => self.0.eval(x),
            (y, x) => {
                let xy = cons(y - 1, x);
                let input = concat(self.eval(xy), xy);
                self.1.eval(input)
            }
        }
    }
}

fn main() {
    let one = Compose(Zero, S);
    dbg!(one.eval([]));
    let add: Rec<1, 1, Proj<0>, Compose<3, 1, 1, Proj<0>, S>> =
        Rec(Proj::<0>, Compose(Proj::<0>, S));
    dbg!(add.eval([3, 2]));
}
