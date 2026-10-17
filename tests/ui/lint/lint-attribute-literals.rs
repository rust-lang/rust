// Checks that lint attributes with literals have an explanatory error

//@ compile-flags: -Zdeduplicate-diagnostics=yes

#[allow("literal")]
//~^ ERROR malformed lint attribute input [E0452]
fn main() {}
