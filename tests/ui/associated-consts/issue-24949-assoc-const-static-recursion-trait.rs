// Check for recursion involving references to trait-associated const.

trait Foo {
    const BAR: u32;
}

const TRAIT_REF_BAR: u32 = <GlobalTraitRef>::BAR;

struct GlobalTraitRef;

impl Foo for GlobalTraitRef {
    const BAR: u32 = TRAIT_REF_BAR; //~ ERROR E0391
}

fn main() {}
