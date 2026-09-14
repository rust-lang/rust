//@ compile-flags: -Z deduplicate-diagnostics=yes

// Trait used as a derive target should point at the trait definition and
// suggest a manual implementation — both when the trait is already in scope
// (via import or local definition) and when it is only importable.

mod inner {
    pub trait MyTrait {} //~ NOTE `MyTrait` is a trait, not a derive macro
    pub trait OuterTrait {} //~ NOTE `OuterTrait` is a trait, not a derive macro
}

use inner::MyTrait;

trait LocalTrait {}
//~^ NOTE `LocalTrait` is a trait, not a derive macro

// in-scope: locally defined
#[derive(LocalTrait)]
//~^ ERROR cannot find derive macro `LocalTrait` in this scope
struct A;

// in-scope: imported
#[derive(MyTrait)]
//~^ ERROR cannot find derive macro `MyTrait` in this scope
struct B;

// out-of-scope: importable but not imported
#[derive(OuterTrait)]
//~^ ERROR cannot find derive macro `OuterTrait` in this scope
struct C;

fn main() {}
