//@ edition:2015
use extern::foo; //~ ERROR expected identifier, found keyword `extern`
                 //~| ERROR unresolved import `r#extern`

fn main() {}
