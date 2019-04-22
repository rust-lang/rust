// Test that structs with higher-ranked where clauses don't generate
// "outlives" requirements. Issue #22246.

// revisions: migrate nll
//[nll]compile-flags: -Z borrowck=mir

#![allow(dead_code)]


///////////////////////////////////////////////////////////////////////////

pub trait TheTrait<'b> {
    type TheAssocType;
}

pub struct TheType<'b> {
    m: [fn(&'b()); 0]
}

impl<'a,'b> TheTrait<'a> for TheType<'b> {
    type TheAssocType = &'b ();
}

///////////////////////////////////////////////////////////////////////////

pub struct WithHrAssoc<T>
    where for<'a> T : TheTrait<'a>
{
    m: [T; 0]
}

fn with_assoc<'a,'b>() {
    // We get an error because 'b:'a does not hold:

    let _: &'a WithHrAssoc<TheType<'b>> = loop { };
    //[migrate]~^ ERROR reference has a longer lifetime
    //[nll]~^^ ERROR lifetime may not live long enough
}

///////////////////////////////////////////////////////////////////////////

pub trait TheSubTrait : for<'a> TheTrait<'a> {
}

impl<'b> TheSubTrait for TheType<'b> { }

pub struct WithHrAssocSub<T>
    where T : TheSubTrait
{
    m: [T; 0]
}

fn with_assoc_sub<'a,'b>() {
    // The error here is just because `'b:'a` must hold for the type
    // below to be well-formed, it is not related to the HR relation.

    let _: &'a WithHrAssocSub<TheType<'b>> = loop { };
    //[migrate]~^ ERROR reference has a longer lifetime
    //[nll]~^^ ERROR lifetime may not live long enough
}


fn main() {
}
