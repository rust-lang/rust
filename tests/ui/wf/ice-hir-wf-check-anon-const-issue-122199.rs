trait Trait<const N: Trait = bar> {
    //~^ ERROR cannot find value `bar` in this scope
    //~| ERROR cycle detected when determining object safety of trait `Trait`
    fn fnc<const N: Trait = u32>(&self) -> Trait {
    //~^ ERROR the name `N` is already used for a generic parameter in this item's generic parameters
    //~| ERROR expected value, found builtin type `u32`
    //~| ERROR defaults for const parameters are only allowed in `struct`, `enum`, `type`, or `trait` definitions
        bar
        //~^ ERROR cannot find value `bar` in this scope
    }
}

fn main() {}
