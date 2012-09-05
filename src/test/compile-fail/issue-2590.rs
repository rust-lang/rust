use dvec::DVec;

type parser = {
    tokens: DVec<int>,
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
