// checks that this attribute is caught on non-macro items.
// this needs a different test since this is done after expansion

#[allow_internal_unstable()] //~ ERROR allow_internal_unstable side-steps
struct S;

fn main() {}
