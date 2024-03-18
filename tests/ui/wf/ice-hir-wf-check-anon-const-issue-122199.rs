trait Trait<const N: Trait = bar> {
//~^ ERROR cannot find value `bar` in this scope
//~| ERROR cycle detected when computing type of `Trait::N`
//~| ERROR cycle detected when computing type of `Trait::N`
//~| ERROR `(dyn Trait<{const error}> + 'static)` is forbidden as the type of a const generic parameter
//~| WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    fn fnc(&self) {
    }
}

fn main() {}
