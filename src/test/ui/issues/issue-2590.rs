struct parser {
    tokens: Vec<isize> ,
}

trait parse {
    fn parse(&self) -> Vec<isize> ;
}

impl parse for parser {
    fn parse(&self) -> Vec<isize> {
        self.tokens //~ ERROR cannot move out of borrowed content
    }
}

fn main() {}
