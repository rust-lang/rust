#![feature(generic_associated_types)]
// check-fail

enum Either<L, R> {
    Left(L),
    Right(R),
}

pub trait HasChildrenOf {
    type T;
    type TRef<'a>;

    fn ref_children<'a>(&'a self) -> Vec<Self::TRef<'a>>;
    fn take_children(self) -> Vec<Self::T>;
}

impl<Left, Right> HasChildrenOf for Either<Left, Right>
where
    Left: HasChildrenOf,
    Right: HasChildrenOf,
{
    type T = Either<Left::T, Right::T>;
    type TRef<'a>
    //~^ the associated type
    //~^^ the associated type
    where
    <Left as HasChildrenOf>::T: 'a,
    <Right as HasChildrenOf>::T: 'a
    = Either<&'a Left::T, &'a Right::T>;

    fn ref_children<'a>(&'a self) -> Vec<Self::TRef<'a>> {
        todo!()
    }

    fn take_children(self) -> Vec<Self::T> {
        todo!()
    }
}

fn main() {}
