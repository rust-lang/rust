//@compile-flags:-D bare_trait_object

//@error-in-other-file:lint `bare_trait_object` has been renamed to `bare_trait_objects`
//@error-in-other-file:requested on the command line with `-D bare_trait_object`
//@error-in-other-file:unused

#[deny(unused)]
fn main() { let unused = (); }
