import dvec::dvec;

type parser = {
    tokens: dvec<int>,
};

trait parse {
    fn parse() -> ~[mut int];
}

impl parser of parse for parser {
    fn parse() -> ~[mut int] {
        dvec::unwrap(self.tokens) //~ ERROR illegal move from self
    }
}

fn main() {}
