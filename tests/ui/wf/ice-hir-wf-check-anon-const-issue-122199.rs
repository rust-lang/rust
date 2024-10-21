trait Trait<const N: Trait = bar> {
    //~^ ERROR cannot find value `bar` in this scope
    //~| ERROR cycle detected when computing type of `Trait::N`
    //~| ERROR the trait `Trait` cannot be made into an object
    //~| ERROR the trait `Trait` cannot be made into an object
    //~| ERROR the trait `Trait` cannot be made into an object
    //~| WARN trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
    //~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //~| WARN trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
    //~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    fn fnc<const N: Trait = u32>(&self) -> Trait {
        //~^ ERROR the name `N` is already used for a generic parameter in this item's generic parameters
        //~| ERROR expected value, found builtin type `u32`
        //~| ERROR defaults for const parameters are only allowed in `struct`, `enum`, `type`, or `trait` definitions
        //~| ERROR associated item referring to unboxed trait object for its own trait
        //~| ERROR the trait `Trait` cannot be made into an object
        //~| WARN trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
        //~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
        //~| WARN trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
        //~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
        //~| WARN trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
        //~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
        bar
        //~^ ERROR cannot find value `bar` in this scope
    }
}

fn main() {}
