// Test that we are imposing the requirement that every associated
// type of a bound that appears in the where clause on a struct must
// outlive the location in which the type appears, even when the
// constraint is in a where clause not a bound. Issue #22246.

#![allow(dead_code)]

pub trait TheTrait {
    type TheAssocType;
}

pub struct TheType<'b> {
    m: [fn(&'b()); 0]
}

impl<'b> TheTrait for TheType<'b> {
    type TheAssocType = &'b ();
}

pub struct WithAssoc<T> where T : TheTrait {
    m: [T; 0]
}

fn with_assoc<'a,'b>() {
    // For this type to be valid, the rules require that all
    // associated types of traits that appear in `WithAssoc` must
    // outlive 'a. In this case, that means TheType<'b>::TheAssocType,
    // which is &'b (), must outlive 'a.

    let _: &'a WithAssoc<TheType<'b>> = loop { };
    //~^ ERROR lifetime may not live long enough
}

fn main() {
}
