// compile-flags:-D renamed-and-removed-lints -D bare_trait_object

// error-pattern:lint `bare_trait_object` has been renamed to `bare_trait_objects`
// error-pattern:use the new name `bare_trait_objects`
// error-pattern:requested on the command line with `-D bare_trait_object`
// error-pattern:requested on the command line with `-D renamed-and-removed-lints`
// error-pattern:unused

#[deny(unused)]
fn main() { let unused = (); }
