trait Trait1 {}
trait Trait2 {}

#[cfg(not_enabled)]
impl Trait1 for .. {}

impl Trait2 for .. {} //~ ERROR `impl Trait for .. {}` is an obsolete syntax

fn main() {}
