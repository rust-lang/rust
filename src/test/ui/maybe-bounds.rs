trait Tr: ?Sized {} //~ ERROR `?Trait` is not permitted in supertraits

type A1 = Tr + (?Sized); //~ ERROR `?Trait` is not permitted in trait object types
type A2 = for<'a> Tr + (?Sized); //~ ERROR `?Trait` is not permitted in trait object types

fn main() {}
