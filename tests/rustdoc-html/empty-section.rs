#![crate_name = "foo"]
#![feature(negative_impls, freeze_impls, freeze, unsafe_unpin)]

pub struct Foo;

//@ has foo/struct.Foo.html
//@ !hasraw - 'Auto Trait Implementations'
// Manually un-implement all auto traits for Foo:
impl !Send for Foo {}
impl !Sync for Foo {}
impl !std::marker::Freeze for Foo {}
impl !std::marker::UnsafeUnpin for Foo {}
impl !std::marker::Unpin for Foo {}
impl !std::panic::RefUnwindSafe for Foo {}
impl !std::panic::UnwindSafe for Foo {}
