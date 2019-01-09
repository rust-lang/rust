macro_rules! e {
    ($inp:ident) => (
        $nonexistent
        //~^ ERROR unknown macro variable `nonexistent`
    );
}

fn main() {
    e!(foo);
}
