// ignore-tidy-linelength

// Checks to make sure that `dyn Trait + Send` and `dyn Trait + Send + Send` are the same type.
// Issue: #47010

struct Struct;

impl Trait for Struct {}

trait Trait {}

type Send1 = Trait + Send;
type Send2 = Trait + Send + Send;
//~^ WARNING duplicate auto trait `Send` found in type parameter bounds [duplicate_auto_traits_in_bounds]

fn main () {}

impl Trait + Send {
    fn test(&self) {
    //~^ ERROR duplicate definitions with name `test`
        println!("one");
    }
}

impl Trait + Send + Send {
//~^ WARNING duplicate auto trait `Send` found in type parameter bounds [duplicate_auto_traits_in_bounds]
    fn test(&self) {
        println!("two");
    }
}
