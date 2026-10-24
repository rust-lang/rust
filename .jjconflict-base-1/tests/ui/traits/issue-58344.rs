//@ check-pass

use std::ops::Add;

trait Trait<T> {
    fn get(self) -> T;
}

struct Holder<T>(T);

impl<T> Trait<T> for Holder<T> {
    fn get(self) -> T {
        self.0
    }
}

enum Either<L, R> {
    Left(L),
    Right(R),
}

impl<L, R> Either<L, R> {
    fn converge<T>(self) -> T
    where
        L: Trait<T>,
        R: Trait<T>,
    {
        match self {
            Either::Left(val) => val.get(),
            Either::Right(val) => val.get(),
        }
    }
}

fn add_generic<A: Add<B>, B>(
    lhs: A,
    rhs: B,
) -> Either<impl Trait<<A as Add<B>>::Output>, impl Trait<<A as Add<B>>::Output>> {
    if true { Either::Left(Holder(lhs + rhs)) } else { Either::Right(Holder(lhs + rhs)) }
}

fn add_one(
    value: u32,
) -> Either<impl Trait<<u32 as Add<u32>>::Output>, impl Trait<<u32 as Add<u32>>::Output>> {
    add_generic(value, 1u32)
}

pub fn main() {
    add_one(3).converge();
}
