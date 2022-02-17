#![feature(negative_impls)]
#![crate_name = "foo"]

pub struct Foo;

// @has foo/struct.Foo.html
// @has - '//*[@class="sidebar-title"]/a[@href="#trait-implementations"]' 'Trait Implementations'
// @has - '//*[@class="sidebar-elems"]//section//a' '!Sync'
impl !Sync for Foo {}
