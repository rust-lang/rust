#![allow(clippy::derive_partial_eq_without_eq)]

#[derive(PartialEq, Hash)]
struct Foo;

impl PartialEq<u64> for Foo {
    fn eq(&self, _: &u64) -> bool {
        true
    }
}

#[derive(Hash)]
//~^ derived_hash_with_manual_eq

struct Bar;

impl PartialEq for Bar {
    fn eq(&self, _: &Bar) -> bool {
        true
    }
}

#[derive(Hash)]
//~^ derived_hash_with_manual_eq

struct Baz;

impl PartialEq<Baz> for Baz {
    fn eq(&self, _: &Baz) -> bool {
        true
    }
}

// Implementing `Hash` with a derived `PartialEq` is fine. See #2627

#[derive(PartialEq)]
struct Bah;

impl std::hash::Hash for Bah {
    fn hash<H: std::hash::Hasher>(&self, _: &mut H) {}
}

fn main() {}
