#![feature(optin_builtin_traits)]
#![crate_name = "foo"]

pub struct Foo;

// @has foo/struct.Foo.html
// @has - '//*[@class="sidebar-title"][@href="#implementations"]' 'Trait Implementations'
// @has - '//*[@class="sidebar-links"]/a' '!Sync'
impl !Sync for Foo {}
