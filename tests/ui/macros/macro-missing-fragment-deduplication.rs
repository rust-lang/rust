//@ compile-flags: -Zdeduplicate-diagnostics=yes

macro_rules! m {
    ($name) => {}; //~ ERROR missing fragment
                   //~| ERROR missing fragment
}

fn main() {
    m!();
    m!();
    m!();
    m!();
}
