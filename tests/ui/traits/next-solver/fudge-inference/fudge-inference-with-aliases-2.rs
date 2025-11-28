//@ compile-flags: -Znext-solver
//@ check-pass

// A regression test for https://github.com/rust-lang/trait-system-refactor-initiative/issues/252.
// `fn fudge_inference_if_ok` might lose relationships between ty vars so we need to normalize
// them inside the fudge scope.

enum Either<L, R> {
    Left(L),
    Right(R),
}
impl<L, R> Iterator for Either<L, R>
where
    L: Iterator,
    R: Iterator<Item = L::Item>,
{
    type Item = L::Item;
    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

pub enum OneOrMany<I: Iterator> {
    One(I::Item),
    Many(I),
}
pub fn repro<T>(iter: impl IntoIterator<Item = T>) -> OneOrMany<impl Iterator<Item = T>> {
    let mut iter = iter.into_iter();
    // if the order of two ifs is reversed: no error
    if true {
        return OneOrMany::Many(Either::Left(iter));
    }
    if let Some(first) = iter.next() {
        return OneOrMany::One(first);
    }

    OneOrMany::Many(Either::Right(iter))
}

fn main() {}
