#![feature(type_alias_impl_trait)]

// Check that, even within the defining scope, we can get the
// `Send` or `Sync` bound of a TAIT, as long as the current function
// doesn't mention those TAITs in the signature
type Foo = impl std::fmt::Debug;
//~^ ERROR cycle detected

fn is_send<T: Send>() {}

fn foo() -> Foo {
    42
}

fn bar() {
    is_send::<Foo>();
}

fn main() {}
