#![crate_name = "foo"]
#![crate_type = "rlib"]

static FOO: uint = 3;

pub fn foo() -> &'static uint { &FOO }
