#![crate_name = "a"]
#![crate_type = "rlib"]

static FOO: uint = 3;

pub fn token() -> &'static uint { &FOO }
