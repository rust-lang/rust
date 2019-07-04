// compile-flags:-Z unstable-options --show-coverage
// build-pass (FIXME(62277): could be check-pass?)

#![feature(trait_alias)]

/// look at this trait right here
pub trait ThisTrait {
    /// that's a trait all right
    fn right_here(&self);

    /// even the provided functions show up as trait methods
    fn aww_yeah(&self) {}

    /// gotta check those associated types, they're slippery
    type SomeType;
}

/// so what happens if we take some struct...
pub struct SomeStruct;

/// ...and slap this trait on it?
impl ThisTrait for SomeStruct {
    /// nothing! trait impls are totally ignored in this calculation, sorry.
    fn right_here(&self) {}

    type SomeType = String;
}

/// but what about those aliases? i hear they're pretty exotic
pub trait MyAlias = ThisTrait + Send + Sync;

// FIXME(58624): once rustdoc can process existential types, we need to make sure they're counted
// /// woah, getting all existential in here
// pub existential type ThisExists: ThisTrait;
//
// /// why don't we get a little more concrete
// pub fn defines() -> ThisExists { SomeStruct {} }
