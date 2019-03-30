// The purpose of this test is to demonstrate that `?Sized` is allowed in trait objects
// (thought it has no effect).

type _0 = dyn ?Sized;
//~^ ERROR at least one non-builtin trait is required for an object type [E0224]

type _1 = dyn Clone + ?Sized;

type _2 = dyn Clone + ?Sized + ?Sized;

type _3 = dyn ?Sized + Clone;

fn main() {}
