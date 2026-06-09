//~ ERROR overflow evaluating the requirement `Self: StreamingIterator<'_>` [E0275]
// Regression test for <https://github.com/rust-lang/rust/issues/153354>.

trait StreamingIterator<'a> {
    type Item: 'a;
}

impl<'b, I, T> StreamingIterator<'b> for I
//~^ ERROR the type parameter `T` is not constrained by the impl trait, self type, or predicates [E0207]
where
    I: IntoIterator,
    T: FnMut(Self::Item, I::Item),
{
    type Item = T;
    //~^ ERROR overflow evaluating the requirement `I: IntoIterator` [E0275]
}

fn main() {}
