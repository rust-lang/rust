#![crate_name = "foo"]

#![feature(optin_builtin_traits)]

pub struct Foo;

// @has foo/struct.Foo.html
// @!has - 'Auto Trait Implementations'
impl !Send for Foo {}
impl !Sync for Foo {}
impl !std::marker::Unpin for Foo {}
impl !std::panic::RefUnwindSafe for Foo {}
impl !std::panic::UnwindSafe for Foo {}
