trait NewTrait : SomeNonExistentTrait {}
//~^ ERROR cannot find trait `SomeNonExistentTrait` in this scope

impl SomeNonExistentTrait for isize {}
//~^ ERROR cannot find trait `SomeNonExistentTrait` in this scope

fn f<T:SomeNonExistentTrait>() {}
//~^ ERROR cannot find trait `SomeNonExistentTrait` in this scope

fn main() {}
