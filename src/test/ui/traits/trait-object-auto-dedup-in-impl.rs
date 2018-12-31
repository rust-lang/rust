// Checks to make sure that `dyn Trait + Send` and `dyn Trait + Send + Send` are the same type.
// See issue #47010.

struct Struct;

impl Trait for Struct {}

trait Trait {}

type Send1 = Trait + Send;
type Send2 = Trait + Send + Send;

fn main () {}

impl Trait + Send {
    fn test(&self) {
    //~^ ERROR duplicate definitions with name `test`
        println!("one");
    }
}

impl Trait + Send + Send {
    fn test(&self) {
        println!("two");
    }
}
