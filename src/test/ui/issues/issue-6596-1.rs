macro_rules! e {
    ($inp:ident) => (
        $nonexistent
        //~^ ERROR unknown macro variable `nonexistent`
        //~| ERROR cannot find value `nonexistent` in this scope
    );
}

fn main() {
    e!(foo);
}
