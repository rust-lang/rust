fn main() {}

type X<'a> = (?'a) +;
//~^ ERROR `?` may only modify trait bounds, not lifetime bounds
//~| ERROR at least one trait is required for an object type
//~| WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition
