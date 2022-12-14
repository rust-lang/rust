fn main() {}

type X<'a> = (?'a) +;
//~^ ERROR `?` may only modify trait bounds, not lifetime bounds
//~| ERROR at least one trait is required for an object type
