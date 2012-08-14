#[forbid(implicit_copies)];

import dvec::dvec;

type parser = {
    tokens: dvec<int>,
};

trait parse {
    fn parse() -> ~[mut int];
}

impl parser: parse {
    fn parse() -> ~[mut int] {
        dvec::unwrap(self.tokens) //~ ERROR implicitly copying
    }
}

fn main() {}
