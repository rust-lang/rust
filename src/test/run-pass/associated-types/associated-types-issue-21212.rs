// run-pass
#![allow(unused_variables)]
// Regression test for #21212: an overflow occurred during trait
// checking where normalizing `Self::Input` led to normalizing the
// where clauses in the environment which in turn required normalizing
// `Self::Input`.


pub trait Parser {
    type Input;

    fn parse(input: <Self as Parser>::Input) {
        panic!()
    }
}

impl <P> Parser for P {
    type Input = ();
}

fn main() {
}
