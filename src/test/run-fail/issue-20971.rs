// Regression test for Issue #20971.

// error-pattern:Hello, world!

pub trait Parser {
    type Input;
    fn parse(&mut self, input: <Self as Parser>::Input);
}

impl Parser for () {
    type Input = ();
    fn parse(&mut self, input: ()) {}
}

pub fn many() -> Box<Parser<Input = <() as Parser>::Input> + 'static> {
    panic!("Hello, world!")
}

fn main() {
    many().parse(());
}
