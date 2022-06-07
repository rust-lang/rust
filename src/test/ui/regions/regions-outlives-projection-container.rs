// Test that we are imposing the requirement that every associated
// type of a bound that appears in the where clause on a struct must
// outlive the location in which the type appears. Issue #22246.

#![allow(dead_code)]
#![feature(rustc_attrs)]

pub trait TheTrait {
    type TheAssocType;
}

pub struct TheType<'b> {
    m: [fn(&'b()); 0]
}

impl<'b> TheTrait for TheType<'b> {
    type TheAssocType = &'b ();
}

pub struct WithAssoc<T:TheTrait> {
    m: [T; 0]
}

pub struct WithoutAssoc<T> {
    m: [T; 0]
}

fn with_assoc<'a,'b>() {
    // For this type to be valid, the rules require that all
    // associated types of traits that appear in `WithAssoc` must
    // outlive 'a. In this case, that means TheType<'b>::TheAssocType,
    // which is &'b (), must outlive 'a.

    // FIXME (#54943) NLL doesn't enforce WF condition in unreachable code if
    // `_x` is changed to `_`
    let _x: &'a WithAssoc<TheType<'b>> = loop { };
    //~^ ERROR lifetime may not live long enough
}

fn with_assoc1<'a,'b>() where 'b : 'a {
    // For this type to be valid, the rules require that all
    // associated types of traits that appear in `WithAssoc` must
    // outlive 'a. In this case, that means TheType<'b>::TheAssocType,
    // which is &'b (), must outlive 'a, so 'b : 'a must hold, and
    // that is in the where clauses, so we're fine.

    let _x: &'a WithAssoc<TheType<'b>> = loop { };
}

fn without_assoc<'a,'b>() {
    // Here there are no associated types but there is a requirement
    // that `'b:'a` holds because the `'b` appears in `TheType<'b>`.

    let _x: &'a WithoutAssoc<TheType<'b>> = loop { };
    //~^ ERROR lifetime may not live long enough
}

fn call_with_assoc<'a,'b>() {
    // As `with_assoc`, but just checking that we impose the same rule
    // on the value supplied for the type argument, even when there is
    // no data.

    call::<&'a WithAssoc<TheType<'b>>>();
    //~^ ERROR lifetime may not live long enough
}

fn call_without_assoc<'a,'b>() {
    // As `without_assoc`, but in a distinct scenario.

    call::<&'a WithoutAssoc<TheType<'b>>>();
    //~^ ERROR lifetime may not live long enough
}

fn call<T>() { }

fn main() {
}
