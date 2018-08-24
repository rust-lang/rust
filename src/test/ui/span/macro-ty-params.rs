macro_rules! m {
    ($p1: path) => {
        #[derive($p1)] struct U;
    }
}

fn main() {
    foo::<T>!(); //~ ERROR generic arguments in macro path
    foo::<>!(); //~ ERROR generic arguments in macro path
    m!(MyTrait<>); //~ ERROR generic arguments in macro path
    //~^ ERROR unexpected generic arguments in path
}
