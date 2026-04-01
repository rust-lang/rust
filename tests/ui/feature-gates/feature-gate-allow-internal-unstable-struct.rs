// checks that this attribute is caught on non-macro items.
// this needs a different test since this is done after expansion

// FIXME(jdonszelmann): empty attributes are currently ignored, since when its empty no actual
// change is applied. This should be fixed when later moving this check to attribute parsing.
#[allow_internal_unstable(something)] //~ ERROR allow_internal_unstable side-steps
//~| ERROR attribute cannot be used on
struct S;

fn main() {}
