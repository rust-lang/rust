//@ compile-flags: -Zdeduplicate-diagnostics=yes

macro_rules! m {
    ($name) => {}; //~ ERROR missing fragment
}

fn main() {
    m!(); //~ ERROR unexpected end
    m!(); //~ ERROR unexpected end
    m!(); //~ ERROR unexpected end
    m!(); //~ ERROR unexpected end
}
