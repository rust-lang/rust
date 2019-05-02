trait Tr: ?Sized {} //~ ERROR `?Trait` is not permitted in supertraits

type A1 = dyn Tr + (?Sized);
type A2 = dyn for<'a> Tr + (?Sized);

fn main() {}
