// Test that `dyn ?Sized` (i.e., a trait object with only a maybe buond) is not allowed.

type _0 = dyn ?Sized;
//~^ ERROR at least one trait is required for an object type [E0224]
//~| ERROR relaxed bounds are not permitted in trait object types

fn main() {}
