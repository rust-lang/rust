#![crate_name = "foo"]
#![feature(negative_impls, freeze_impls, freeze, unsafe_unpin, move_trait, min_specialization)]
// FIXME(move-trait): remove this when it's complete
#![expect(incomplete_features)]

pub struct Foo;

//@ has foo/struct.Foo.html
//@ !hasraw - 'Auto Trait Implementations'
// Manually un-implement all auto traits for Foo:
impl !std::marker::Move for Foo {}
impl !Send for Foo {}
impl !Sync for Foo {}
impl !std::marker::Freeze for Foo {}
impl !std::marker::UnsafeUnpin for Foo {}
impl !std::marker::Unpin for Foo {}
impl !std::panic::RefUnwindSafe for Foo {}
impl !std::panic::UnwindSafe for Foo {}
