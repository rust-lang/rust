import dvec::dvec;

type parser = {
    tokens: dvec<int>,
};

trait parse {
    fn parse() -> ~[mut int];
}

impl parser: parse {
    fn parse() -> ~[mut int] {
        dvec::unwrap(move self.tokens) //~ ERROR illegal move from self
    }
}

fn main() {}
