//@ check-pass

/// [`…foo`] //~ WARN: unresolved link
/// [`…bar`] //~ WARN: unresolved link
/// [`Err`]
pub struct Broken {}

/// [`…`] //~ WARN: unresolved link
/// [`…`] //~ WARN: unresolved link
/// [`Err`]
pub struct Broken2 {}

/// [`…`][…] //~ WARN: unresolved link
/// [`…`][…] //~ WARN: unresolved link
/// [`Err`]
pub struct Broken3 {}

/// […………………………][Broken3]
pub struct Broken4 {}

/// [Broken3][…………………………] //~ WARN: unresolved link
pub struct Broken5 {}

pub struct Err;
