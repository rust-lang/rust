// Test that relaxed `Sized` bounds are rejected in trait object types.

trait Foo {}

type _0 = dyn ?Sized + Foo;
//~^ ERROR relaxed bounds are not permitted in trait object types

type _1 = dyn Foo + ?Sized;
//~^ ERROR relaxed bounds are not permitted in trait object types

type _2 = dyn Foo + ?Sized + ?Sized;
//~^ ERROR relaxed bounds are not permitted in trait object types
//~| ERROR relaxed bounds are not permitted in trait object types

type _3 = dyn ?Sized + Foo;
//~^ ERROR relaxed bounds are not permitted in trait object types

fn main() {}
