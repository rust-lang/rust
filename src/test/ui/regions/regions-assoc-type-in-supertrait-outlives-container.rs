// Test that we are imposing the requirement that every associated
// type of a bound that appears in the where clause on a struct must
// outlive the location in which the type appears, even when the
// associted type is in a supertype. Issue #22246.

// revisions: migrate nll
//[nll]compile-flags: -Z borrowck=mir

// Since we are testing nll (and migration) explicitly as a separate
// revisions, don't worry about the --compare-mode=nll on this test.

// ignore-compare-mode-nll

#![allow(dead_code)]

pub trait TheTrait {
    type TheAssocType;
}

pub trait TheSubTrait : TheTrait {
}

pub struct TheType<'b> {
    m: [fn(&'b()); 0]
}

impl<'b> TheTrait for TheType<'b> {
    type TheAssocType = &'b ();
}

impl<'b> TheSubTrait for TheType<'b> {
}

pub struct WithAssoc<T:TheSubTrait> {
    m: [T; 0]
}

fn with_assoc<'a,'b>() {
    // For this type to be valid, the rules require that all
    // associated types of traits that appear in `WithAssoc` must
    // outlive 'a. In this case, that means TheType<'b>::TheAssocType,
    // which is &'b (), must outlive 'a.

    let _: &'a WithAssoc<TheType<'b>> = loop { };
    //[migrate]~^ ERROR reference has a longer lifetime
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() {
}
