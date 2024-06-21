trait A {
    type Bla;
    fn to_bla(&self) -> Bla;
    //~^ ERROR cannot find type `Bla`
}

fn main() {}
