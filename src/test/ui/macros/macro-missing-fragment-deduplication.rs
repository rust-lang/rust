// compile-flags: -Zdeduplicate-diagnostics=yes

macro_rules! m {
    ($name) => {}
    //~^ ERROR missing fragment
    //~| ERROR missing fragment
    //~| WARN this was previously accepted
}

fn main() {
    m!();
    m!();
    m!();
    m!();
}
