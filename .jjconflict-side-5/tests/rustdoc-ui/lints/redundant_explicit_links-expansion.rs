// This is a regression test for <https://github.com/rust-lang/rust/issues/141553>.
// If the link is generated from expansion, we should not emit the lint.

#![deny(rustdoc::redundant_explicit_links)]

macro_rules! mac1 {
    () => {
        "provided by a [`BufferProvider`](crate::BufferProvider)."
    };
}

macro_rules! mac2 {
    () => {
        #[doc = mac1!()]
        pub struct BufferProvider;
    }
}

macro_rules! mac3 {
    () => {
        "Provided by"
    };
}

// Should not lint.
#[doc = mac1!()]
pub struct Foo;

// Should not lint.
mac2!{}

#[doc = "provided by a [`BufferProvider`](crate::BufferProvider)."]
/// bla
//~^^ ERROR: redundant_explicit_links
pub struct Bla;

#[doc = mac3!()]
/// a [`BufferProvider`](crate::BufferProvider).
//~^ ERROR: redundant_explicit_links
pub fn f() {}
