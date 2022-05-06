#![feature(marker_trait_attr, negative_impls)]
#[marker]
trait Trait {}

impl !Trait for () {} //~ ERROR negative impls are not allowed for marker traits

struct Local;
impl !Trait for Local {} //~ ERROR negative impls are not allowed for marker traits

fn main() {}
