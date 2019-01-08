macro_rules! m {
    ($p1: path) => {
        #[derive($p1)] struct U;
    }
}

macro_rules! foo { () => () }

fn main() {
    foo::<T>!(); //~ ERROR generic arguments in macro path
    foo::<>!(); //~ ERROR generic arguments in macro path
    m!(Default<>); //~ ERROR generic arguments in macro path
    //~^ ERROR unexpected generic arguments in path
}
