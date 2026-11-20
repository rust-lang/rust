#![crate_name = "foo"]

// tidy-alphabetical-start
#![feature(forget_trait)]
#![feature(freeze)]
#![feature(freeze_impls)]
#![feature(min_specialization)]
#![feature(move_trait)]
#![feature(negative_impls)]
#![feature(unsafe_unpin)]
// tidy-alphabetical-end

// FIXME(move-trait): remove this when it's complete
#![expect(incomplete_features)]

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
impl !std::marker::Forget for Foo {}
impl !std::marker::Move for Foo {}
