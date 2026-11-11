// Ensure that empty doc comments don't panic.

//@ check-pass

/*!
*/

///
///
pub struct Foo;

#[doc = "
"]
pub mod Mod {
   //!
   //!
}

/**
*/
pub mod Another {
   #![doc = "
"]
}
