// Regression test for issue #83190, triggering an ICE in borrowck.

//@ check-pass

pub trait Any {}
impl<T> Any for T {}

pub trait StreamOnce {
    type Range;
}

pub trait Parser<Input>: Sized {
    type Output;
    type PartialState;
    fn map(self) -> Map<Self> {
        todo!()
    }
}

pub struct Map<P>(P);
impl<I, P: Parser<I, Output = ()>> Parser<I> for Map<P> {
    type Output = ();
    type PartialState = P::PartialState;
}

struct TakeWhile1<Input>(Input);
impl<I: StreamOnce> Parser<I> for TakeWhile1<I> {
    type Output = I::Range;
    type PartialState = ();
}
impl<I> TakeWhile1<I> {
    fn new() -> Self {
        todo!()
    }
}

impl<I, A: Parser<I>> Parser<I> for (A,) {
    type Output = ();
    type PartialState = Map<A::Output>;
}

pub fn metric_stream_parser<'a, I>() -> impl Parser<I, Output = (), PartialState = impl Any + 'a>
where
    I: StreamOnce<Range = &'a [()]>,
{
    (TakeWhile1::new(),).map()
}

fn main() {}
