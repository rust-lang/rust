// Checks that lint attributes with literals have an explanatory error

#[allow("literal")]
//~^ ERROR expected a lint, found literal
//~| ERROR expected a lint, found literal
//~| ERROR expected a lint, found literal
fn main() {}
