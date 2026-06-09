#![warn(clippy::manual_non_exhaustive)]
#![allow(unused)]
//@no-rustfix
pub enum E {
    //~^ manual_non_exhaustive
    A,
    B,
    #[doc(hidden)]
    _C,
}

// if the user explicitly marks as nonexhaustive we shouldn't warn them
#[non_exhaustive]
pub enum Ep {
    A,
    B,
    #[doc(hidden)]
    _C,
}

// marker variant does not have doc hidden attribute, should be ignored
pub enum NoDocHidden {
    A,
    B,
    _C,
}

// name of variant with doc hidden does not start with underscore
pub enum NoUnderscore {
    //~^ manual_non_exhaustive
    A,
    B,
    #[doc(hidden)]
    C,
}

// variant with doc hidden is not unit, should be ignored
pub enum NotUnit {
    A,
    B,
    #[doc(hidden)]
    _C(bool),
}

// variant with doc hidden is the only one, should be ignored
pub enum OnlyMarker {
    #[doc(hidden)]
    _A,
}

// variant with multiple markers, should be ignored
pub enum MultipleMarkers {
    A,
    #[doc(hidden)]
    _B,
    #[doc(hidden)]
    _C,
}

// already non_exhaustive and no markers, should be ignored
#[non_exhaustive]
pub enum NonExhaustive {
    A,
    B,
}

// marked is used, don't lint
pub enum UsedHidden {
    #[doc(hidden)]
    _A,
    B,
    C,
}
fn foo(x: &mut UsedHidden) {
    if matches!(*x, UsedHidden::B) {
        *x = UsedHidden::_A;
    }
}

#[expect(clippy::manual_non_exhaustive)]
pub enum ExpectLint {
    A,
    B,
    #[doc(hidden)]
    _C,
}
