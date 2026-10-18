// Checks that lint attributes with literals have an explanatory error

#[allow("literal")]
//~^ ERROR expected path to a lint, found literal
//~| ERROR expected path to a lint, found literal
//~| ERROR expected path to a lint, found literal
fn main() {}
