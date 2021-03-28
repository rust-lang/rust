//! The point of this crate is to test the insensitive case handling.

#![crate_name = "insensitive_case_docs"]

#![allow(non_camel_case_types)]

/// This is ab.
pub struct ab;

impl ab {
    pub fn foo(&self) {}
}

/// This is another Ab!
pub struct Ab;

impl Ab {
    pub fn bar(&self) {}
}
