// Checks that lint attributes with literals have an explanatory error

//@ compile-flags: -Zdeduplicate-diagnostics=yes

#[allow("literal")]
//~^ ERROR expected path to a lint, found literal
fn main() {}
