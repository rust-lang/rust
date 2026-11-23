// Checks that lint attributes with literals have an explanatory error

#[allow("literal")]
//~^ ERROR malformed
#[deny(1.0)]
//~^ ERROR malformed
fn main() {}
