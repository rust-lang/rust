// Regression test for Issue #20971.

//@ run-fail
//@ error-pattern:Hello, world!
//@ ignore-emscripten no processes

pub trait Parser {
    type Input;
    fn parse(&mut self, input: <Self as Parser>::Input);
}

impl Parser for () {
    type Input = ();
    fn parse(&mut self, input: ()) {}
}

pub fn many() -> Box<dyn Parser<Input = <() as Parser>::Input> + 'static> {
    panic!("Hello, world!")
}

fn main() {
    many().parse(());
}
