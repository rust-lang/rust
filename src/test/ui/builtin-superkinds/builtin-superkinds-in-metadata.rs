// aux-build:trait_superkinds_in_metadata.rs

// Test for traits inheriting from the builtin kinds cross-crate.
// Mostly tests correctness of metadata.

extern crate trait_superkinds_in_metadata;
use trait_superkinds_in_metadata::{RequiresRequiresShareAndSend, RequiresShare};

struct X<T>(T);

impl <T:Sync> RequiresShare for X<T> { }

impl <T:Sync+'static> RequiresRequiresShareAndSend for X<T> { }
//~^ ERROR `T` cannot be sent between threads safely [E0277]

fn main() { }
