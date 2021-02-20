// check-pass

#![feature(control_flow_enum)]
#![feature(try_trait_v2)]

use std::ops::{ControlFlow, Try, FromResidual};

pub fn simple_fold<A, T>(
    iter: impl Iterator<Item = T>,
    mut accum: A,
    mut f: impl FnMut(A, T) -> A,
) -> A {
    for x in iter {
        accum = f(accum, x);
    }
    accum
}

pub fn simple_try_fold_1<A, T, R: Try<Ok = A>>(
    iter: impl Iterator<Item = T>,
    mut accum: A,
    mut f: impl FnMut(A, T) -> R,
) -> R {
    todo!()
}

pub fn simple_try_fold_2<A, T, R: Try<Ok = A>>(
    iter: impl Iterator<Item = T>,
    mut accum: A,
    mut f: impl FnMut(A, T) -> R,
) -> R {
    for x in iter {
        let cf = f(accum, x).branch();
        match cf {
            ControlFlow::Continue(a) => accum = a,
            ControlFlow::Break(_) => todo!(),
        }
    }
    R::from_output(accum)
}

pub fn simple_try_fold_3<A, T, R: Try<Ok = A>>(
    iter: impl Iterator<Item = T>,
    mut accum: A,
    mut f: impl FnMut(A, T) -> R,
) -> R {
    for x in iter {
        let cf = f(accum, x).branch();
        match cf {
            ControlFlow::Continue(a) => accum = a,
            ControlFlow::Break(h) => return R::from_residual(h),
        }
    }
    R::from_output(accum)
}

fn simple_try_fold<A, T, R: Try<Ok = A>>(
    iter: impl Iterator<Item = T>,
    mut accum: A,
    mut f: impl FnMut(A, T) -> R,
) -> R {
    for x in iter {
        accum = f(accum, x)?;
    }
    R::from_output(accum)
}

fn main() {}
