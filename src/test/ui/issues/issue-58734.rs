trait Trait {
    fn exists(self) -> ();

    fn not_object_safe() -> Self;
}

impl Trait for () {
    fn exists(self) -> () {
    }

    fn not_object_safe() -> Self {
        ()
    }
}

fn main() {
    // object-safe or not, this call is OK
    Trait::exists(());
    // no object safety error
    Trait::nonexistent(());
    //~^ ERROR no function or associated item named `nonexistent` found
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
}
