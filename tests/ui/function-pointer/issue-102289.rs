//@ check-pass

pub(crate) trait Parser: Sized {
    type Output;
    fn parse(&mut self, _input: &str) -> Result<(), ()> {
        loop {}
    }
    fn map<F, B>(self, _f: F) -> Map<Self, F>
    where
        F: FnMut(Self::Output) -> B,
    {
        todo!()
    }
}

pub(crate) struct Chainl1<P, Op>(P, Op);
impl<P, Op> Parser for Chainl1<P, Op>
where
    P: Parser,
    Op: Parser,
    Op::Output: FnOnce(P::Output, P::Output) -> P::Output,
{
    type Output = P::Output;
}
pub(crate) fn chainl1<P, Op>(_parser: P, _op: Op) -> Chainl1<P, Op>
where
    P: Parser,
    Op: Parser,
    Op::Output: FnOnce(P::Output, P::Output) -> P::Output,
{
    loop {}
}

pub(crate) struct Map<P, F>(P, F);
impl<A, B, P, F> Parser for Map<P, F>
where
    P: Parser<Output = A>,
    F: FnMut(A) -> B,
{
    type Output = B;
}

impl Parser for u32 {
    type Output = ();
}

pub fn chainl1_error_consume() {
    fn first<T, U>(t: T, _: U) -> T {
        t
    }
    let _ = chainl1(1, 1.map(|_| first)).parse("");
}

fn main() {}
