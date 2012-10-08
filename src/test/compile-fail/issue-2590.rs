use dvec::DVec;

type parser = {
    tokens: DVec<int>,
};

trait parse {
    fn parse() -> ~[int];
}

impl parser: parse {
    fn parse() -> ~[int] {
        dvec::unwrap(move self.tokens) //~ ERROR illegal move from self
    }
}

fn main() {}
