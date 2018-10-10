#![allow(dead_code)]
#![deny(unused_attributes)]

#[must_use]
fn truth() {}

#[must_use]
fn pain() -> () {}

#[must_use]
fn dignity() -> ! { panic!("despair"); }

struct Fear{}

impl Fear {
    #[must_use] fn bravery() {}
}

trait Suspicion {
    #[must_use] fn inspect();
}

impl Suspicion for Fear {
    // FIXME: it's actually rather problematic for this to get the unused-attributes
    // lint on account of the return type—the unused-attributes lint should fire
    // here, but it should be because `#[must_use]` needs to go on the trait
    // definition, not the impl (Issue #48486)
    #[must_use] fn inspect() {}
}

fn main() {}
