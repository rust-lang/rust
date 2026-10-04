//@ compile-flags: --crate-type lib

// Valid use and extern crate targets
#[doc(inline)]
pub use std::vec::Vec;

#[doc(no_inline)]
pub use std::string::String;

#[doc(inline)]
pub extern crate std as my_std;

#[doc(no_inline)]
pub extern crate core as my_core;

// Repeated identical values
#[doc(inline)]
#[doc(inline)]
pub use std::option::Option;

#[doc(no_inline)]
#[doc(no_inline)]
pub use std::result::Result;

#[doc(inline, inline)]
pub use std::panic;

// Conflicting values in both orders across separate attributes
#[doc(inline)]
//~^ ERROR conflicting doc inlining attributes
//~| HELP remove one of the conflicting attributes
#[doc(no_inline)]
pub use std::fmt;

#[doc(no_inline)]
//~^ ERROR conflicting doc inlining attributes
//~| HELP remove one of the conflicting attributes
#[doc(inline)]
pub use std::io;

// Conflicts within one doc list
#[doc(inline, no_inline)]
//~^ ERROR conflicting doc inlining attributes
//~| HELP remove one of the conflicting attributes
pub use std::fs;

// Invalid targets with relevant lint levels
#[deny(invalid_doc_attributes)]
#[doc(inline)]
//~^ ERROR this attribute can only be applied to a `use` item
pub fn fn_with_inline() {}

#[allow(invalid_doc_attributes)]
#[doc(inline)]
pub fn fn_allowed() {}

#[deny(invalid_doc_attributes)]
#[doc(no_inline)]
//~^ ERROR this attribute can only be applied to a `use` item
pub struct StructWithNoInline;

// Conflict on an invalid target, preserving diagnostic precedence
#[deny(invalid_doc_attributes)]
#[doc(inline)]
//~^ ERROR conflicting doc inlining attributes
//~| HELP remove one of the conflicting attributes
#[doc(no_inline)]
pub struct ConflictOnStruct;

// Malformed arguments without extra validation diagnostics
#[doc(inline = "foo")]
//~^ WARN didn't expect any arguments here
//~| WARN this was previously accepted
pub fn malformed_fn() {}
