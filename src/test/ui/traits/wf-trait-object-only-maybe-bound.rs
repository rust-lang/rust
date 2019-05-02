// Test that `dyn ?Sized` (i.e., a trait object with only a maybe buond) is not allowed.

type _0 = dyn ?Sized;
//~^ ERROR at least one non-builtin trait is required for an object type [E0224]

fn main() {}
