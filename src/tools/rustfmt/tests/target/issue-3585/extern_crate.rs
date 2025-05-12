// rustfmt-inline_attribute_width: 100

#[macro_use] extern crate static_assertions;

#[cfg(unix)] extern crate static_assertions;

// a comment before the attribute
#[macro_use]
// some comment after
extern crate static_assertions;
