#![crate_type = "lib"]
extern crate core;

// Known accidental stabilizations with no known users, slated for un-stabilization
// fully stable @ core::char::UNICODE_VERSION
use core::unicode::UNICODE_VERSION; //~ ERROR use of unstable library feature `unicode_internals`

// Known accidental stabilizations with known users
// fully stable @ core::mem::transmute
use core::intrinsics::transmute; // depended upon by rand_core
//~^WARN deprecated
