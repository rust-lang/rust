//@ run-pass


fn main() {
    let _ = test(Some(0).into_iter());
}

trait Parser {
    type Input: Iterator;
    type Output;
    fn parse(self, input: Self::Input) -> Result<(Self::Output, Self::Input), ()>;
    fn chain<P>(self, p: P) -> Chain<Self, P> where Self: Sized {
        Chain(self, p)
    }
}

struct Token<T>(#[allow(dead_code)] T::Item) where T: Iterator;

impl<T> Parser for Token<T> where T: Iterator {
    type Input = T;
    type Output = T::Item;
    fn parse(self, _input: Self::Input) -> Result<(Self::Output, Self::Input), ()> {
        Err(())
    }
}

struct Chain<L, R>(#[allow(dead_code)] L, #[allow(dead_code)] R);

impl<L, R> Parser for Chain<L, R> where L: Parser, R: Parser<Input = L::Input> {
    type Input = L::Input;
    type Output = (L::Output, R::Output);
    fn parse(self, _input: Self::Input) -> Result<(Self::Output, Self::Input), ()> {
        Err(())
    }
}

fn test<I>(i: I) -> Result<((), I), ()> where I: Iterator<Item = i32> {
    Chain(Token(0), Token(1))
        .chain(Chain(Token(0), Token(1)))
        .chain(Chain(Token(0), Token(1)))
        .chain(Chain(Token(0), Token(1)))
        .chain(Chain(Token(0), Token(1)))
        .chain(Chain(Token(0), Token(1)))
        .chain(Chain(Token(0), Token(1)))
        .chain(Chain(Token(0), Token(1)))
        .chain(Chain(Token(0), Token(1)))
        .parse(i)
        .map(|(_, i)| ((), i))
}
