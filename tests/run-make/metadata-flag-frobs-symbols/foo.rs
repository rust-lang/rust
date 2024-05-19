#![crate_name = "foo"]
#![crate_type = "rlib"]

static FOO: usize = 3;

pub fn foo() -> &'static usize { &FOO }
