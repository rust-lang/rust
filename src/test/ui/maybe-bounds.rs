trait Tr: ?Sized {} //~ ERROR `?Trait` is not permitted in supertraits

type A1 = Tr + (?Sized);
type A2 = for<'a> Tr + (?Sized);

fn main() {}
