trait Tr: ?Sized {}
//~^ ERROR relaxed bounds are not permitted in supertrait bounds

type A1 = dyn Tr + (?Sized);
//~^ ERROR relaxed bounds are not permitted in trait object types
type A2 = dyn for<'a> Tr + (?Sized);
//~^ ERROR relaxed bounds are not permitted in trait object types

fn main() {}
