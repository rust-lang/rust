pub enum Enum {
    A(usize),
}

impl Enum {
    fn foo(&self) -> () {
        match self {
            Self::A => (),
            //~^ ERROR expected unit struct/variant or constant, found tuple variant
        }
    }
}

fn main() {}
