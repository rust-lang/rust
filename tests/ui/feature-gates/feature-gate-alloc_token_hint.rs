#![crate_type = "lib"]

#[alloc_token_hint(contains_pointers = false)] //~ ERROR the `alloc_token_hint` attribute is an experimental feature [E0658]
pub struct Foo(u64);
