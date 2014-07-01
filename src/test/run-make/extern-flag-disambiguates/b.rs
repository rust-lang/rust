#![crate_name = "b"]
#![crate_type = "rlib"]

extern crate a;

static FOO: uint = 3;

pub fn token() -> &'static uint { &FOO }
pub fn a_token() -> &'static uint { a::token() }
