#![feature(macro_rules)]

macro_rules! g {
    ($inp:ident) => (
        { $inp $nonexistent }
        //~^ ERROR unknown macro variable `nonexistent`
        //~| ERROR expected one of
    );
}

fn main() {
    g!(foo);
}
