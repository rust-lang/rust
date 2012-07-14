import dvec::dvec;

type parser = {
    tokens: dvec<int>,
};

impl parser for parser {
    fn parse() -> ~[mut int] {
        dvec::unwrap(self.tokens) //~ ERROR illegal move from self
    }
}

fn main() {}
