// https://github.com/rust-lang/rust/issues/2590
// Tests that you cannot move out of a borrowed field in a struct.
struct Parser {
    tokens: Vec<isize> ,
}

trait Parse {
    fn parse(&self) -> Vec<isize> ;
}

impl Parse for Parser {
    fn parse(&self) -> Vec<isize> {
        self.tokens //~ ERROR cannot move out
    }
}

fn main() {}
