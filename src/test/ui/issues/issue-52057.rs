// Regression test for #52057. There is an implied bound
// that `I: 'a` where `'a` is the lifetime of `self` in `parse_first`;
// but to observe that, one must normalize first.
//
// run-pass

pub trait Parser {
    type Input;

    #[inline(always)]
    fn parse_first(input: &mut Self::Input);
}

impl<'a, I, P: ?Sized> Parser for &'a mut P
where
    P: Parser<Input = I>,
{
    type Input = I;

    fn parse_first(_: &mut Self::Input) {}
}

fn main() {}
