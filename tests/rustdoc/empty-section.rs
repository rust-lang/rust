#![crate_name = "foo"]
#![feature(negative_impls, freeze_impls, freeze)]

pub struct Foo;

//@ has foo/struct.Foo.html
//@ !hasraw - 'Auto Trait Implementations'
impl !Send for Foo {}
impl !Sync for Foo {}
impl !std::marker::Freeze for Foo {}
impl !std::marker::Unpin for Foo {}
impl !std::panic::RefUnwindSafe for Foo {}
impl !std::panic::UnwindSafe for Foo {}
