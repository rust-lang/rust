#![feature(negative_impls)]

// Check that the issue #33140 hack does not allow unintended things.

// OK
trait Trait0 {}

impl Trait0 for dyn Send {}
impl Trait0 for dyn Send {}
//~^ ERROR: E0119

// Problem 1: associated types
trait Trait1 {
    fn my_fn(&self) {}
}

impl Trait1 for dyn Send {}
impl Trait1 for dyn Send {}
//~^ ERROR E0119

// Problem 2: negative impl
trait Trait2 {}

impl Trait2 for dyn Send {}
impl !Trait2 for dyn Send {}
//~^ ERROR E0751

// Problem 3: type parameter
trait Trait3<T: ?Sized> {}

impl Trait3<dyn Sync> for dyn Send {}
impl Trait3<dyn Sync> for dyn Send {}
//~^ ERROR E0119

// Problem 4a: not a trait object - generic
trait Trait4a {}

impl<T: ?Sized> Trait4a for T {}
impl Trait4a for dyn Send {}
//~^ ERROR E0119

// Problem 4b: not a trait object - misc
trait Trait4b {}

impl Trait4b for () {}
impl Trait4b for () {}
//~^ ERROR E0119

// Problem 4c: not a principal-less trait object
trait Trait4c {}

impl Trait4c for dyn Trait1 + Send {}
impl Trait4c for dyn Trait1 + Send {}
//~^ ERROR E0119

// Problem 4d: lifetimes
trait Trait4d {}

impl<'a> Trait4d for dyn Send + 'a {}
impl<'a> Trait4d for dyn Send + 'a {}
//~^ ERROR E0119

// Problem 5: where-clauses
trait Trait5 {}

impl Trait5 for dyn Send {}
impl Trait5 for dyn Send where u32: Copy {}
//~^ ERROR E0119

fn main() {}
