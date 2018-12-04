// ignore-tidy-linelength

// Checks to make sure that `dyn Trait + Send` and `dyn Trait + Send + Send` are the same type.
// Issue: #47010

struct Struct;

impl Trait for Struct {}

trait Trait {}

type Send1 = Trait + Send;
type Send2 = Trait + Send + Send;
//~^ WARNING duplicate auto trait `std::marker::Send` found in trait object [duplicate_auto_traits_in_trait_objects]
//~| this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

fn main () {}

impl Trait + Send {
    fn test(&self) { println!("one"); } //~ ERROR duplicate definitions with name `test`
}

impl Trait + Send + Send {
//~^ WARNING duplicate auto trait `std::marker::Send` found in trait object [duplicate_auto_traits_in_trait_objects]
//~| this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    fn test(&self) { println!("two"); }
}
