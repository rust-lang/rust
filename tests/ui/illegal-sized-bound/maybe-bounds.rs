trait Tr: ?Sized {}
//~^ ERROR `?Trait` is not permitted in supertraits

type A1 = dyn Tr + (?Sized);
//~^ ERROR `?Trait` is not permitted in trait object types
type A2 = dyn for<'a> Tr + (?Sized);
//~^ ERROR `?Trait` is not permitted in trait object types

fn main() {}
