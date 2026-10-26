// Checks that lint attributes with literals have an explanatory error

#[allow("literal")]
//~^ ERROR expected a lint
//~| ERROR expected a lint
//~| ERROR expected a lint
#[deny(1.0)]
//~^ ERROR expected a lint
//~| ERROR expected a lint
//~| ERROR expected a lint
fn main() {}
