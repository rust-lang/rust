//@ check-pass

/// [`…foo`] [`…bar`] [`Err`]
pub struct Broken {}

/// [`…`] [`…`] [`Err`]
pub struct Broken2 {}

/// [`…`][…] [`…`][…] [`Err`]
pub struct Broken3 {}

/// […………………………][Broken3]
pub struct Broken4 {}

/// [Broken3][…………………………]
pub struct Broken5 {}

pub struct Err;
