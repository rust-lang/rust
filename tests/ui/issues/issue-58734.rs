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
    //~^ ERROR no function or associated item named `nonexistent` found
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
}
