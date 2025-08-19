trait Trait<const N: dyn Trait = bar> {
    //~^ ERROR cannot find value `bar` in this scope
    //~| ERROR cycle detected when computing type of `Trait::N`
    fn fnc<const N: dyn Trait = u32>(&self) -> dyn Trait {
        //~^ ERROR the name `N` is already used for a generic parameter in this item's generic parameters
        //~| ERROR expected value, found builtin type `u32`
        //~| ERROR defaults for generic parameters are not allowed here
        bar
        //~^ ERROR cannot find value `bar` in this scope
    }
}

fn main() {}
