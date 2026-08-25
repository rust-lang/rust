//@ compile-flags:-D renamed-and-removed-lints -D bare_trait_object
//@ dont-require-annotations: HELP
//@ dont-require-annotations: NOTE

#[deny(unused)]
fn main() { let unused = (); } //~ ERROR unused variable: `unused`

//~? ERROR lint `bare_trait_object` has been renamed to `bare_trait_objects`
//~? ERROR lint `bare_trait_object` has been renamed to `bare_trait_objects`
//~? ERROR lint `bare_trait_object` has been renamed to `bare_trait_objects`
//~? HELP use the new name `bare_trait_objects`
//~? NOTE requested on the command line with `-D bare_trait_object`
//~? NOTE requested on the command line with `-D renamed-and-removed-lints`
