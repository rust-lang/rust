trait NewTrait : SomeNonExistentTrait {}
//~^ ERROR cannot find trait `SomeNonExistentTrait`

impl SomeNonExistentTrait for isize {}
//~^ ERROR cannot find trait `SomeNonExistentTrait`

fn f<T:SomeNonExistentTrait>() {}
//~^ ERROR cannot find trait `SomeNonExistentTrait`

fn main() {}
