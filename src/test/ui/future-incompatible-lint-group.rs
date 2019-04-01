#![deny(future_incompatible)]

trait Tr {
    fn f(u8) {} //~ ERROR anonymous parameters are deprecated
                //~^ WARN this was previously accepted
                //~| WARN hard error
}

fn main() {}
