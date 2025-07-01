//! Verifies that the reserved underscore `_` cannot be used as an `ident` fragment specifier
//! within a macro pattern, as it leads to a compilation error.

macro_rules! identity {
    ($i: ident) => {
        $i
    };
}

fn main() {
    let identity!(_) = 10; //~ ERROR no rules expected reserved identifier `_`
}
