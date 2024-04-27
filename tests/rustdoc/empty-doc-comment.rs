// Ensure that empty doc comments don't panic.

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
