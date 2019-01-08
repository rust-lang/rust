#![feature(macro_rules)]

macro_rules! g {
    ($inp:ident) => (
        { $inp $nonexistent }
        //~^ ERROR unknown macro variable `nonexistent`
    );
}

fn main() {
    let foo = 0;
    g!(foo);
}
