#[deny(bare_trait_object)]
//~^ WARN lint `bare_trait_object` has been renamed to `bare_trait_objects`
#[deny(unused)]
fn main() { let unused = (); } //~ ERROR unused
