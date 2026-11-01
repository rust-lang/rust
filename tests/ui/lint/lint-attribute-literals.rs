// Checks that lint attributes with literals have an explanatory error

#[allow("literal")]
//~^ ERROR malformed
//~| ERROR malformed
//~| ERROR malformed
#[deny(1.0)]
//~^ ERROR malformed
//~| ERROR malformed
//~| ERROR malformed
fn main() {}
