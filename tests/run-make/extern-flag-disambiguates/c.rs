#![crate_name = "c"]
#![crate_type = "rlib"]

extern crate a;

static FOO: usize = 3;

pub fn token() -> &'static usize { &FOO }
pub fn a_token() -> &'static usize { a::token() }
