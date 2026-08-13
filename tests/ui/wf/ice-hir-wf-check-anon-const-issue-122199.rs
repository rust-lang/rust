trait Trait<const N: dyn Trait = bar> {
    //~^ ERROR cannot find value `bar` in this scope
    //~| ERROR cycle detected when computing type of `Trait::N`
    fn fnc<const N: dyn Trait = u32>(&self) -> dyn Trait {
        //~^ ERROR the name `N` is already used for a generic parameter in this item's generic parameters
        //~| ERROR cannot find value `u32` in this scope
        bar
        //~^ ERROR cannot find value `bar` in this scope
    }
}

fn main() {}
