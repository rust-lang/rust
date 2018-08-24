// compile-flags:-D bare_trait_object

// error-pattern:lint `bare_trait_object` has been renamed to `bare_trait_objects`
// error-pattern:requested on the command line with `-D bare_trait_object`
// error-pattern:unused

#[deny(unused)]
fn main() { let unused = (); }
