#![crate_name = "a"]
#![crate_type = "rlib"]

static FOO: usize = 3;

pub fn token() -> &'static usize {
    &FOO
}
