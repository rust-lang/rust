trait Trait {
    fn exists(self) -> ();

    fn dyn_incompatible() -> Self;
}

impl Trait for () {
    fn exists(self) -> () {
    }

    fn dyn_incompatible() -> Self {
        ()
    }
}

fn main() {
    // dyn-compatible or not, this call is OK
    Trait::exists(());
    // no dyn-compatibility error
    Trait::nonexistent(());
    //~^ WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
    //~| ERROR the trait `Trait` is not dyn compatible
}
