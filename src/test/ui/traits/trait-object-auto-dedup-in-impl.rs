// Checks to make sure that `dyn Trait + Send` and `dyn Trait + Send + Send` are the same type.
// Issue: #47010

struct Struct;
impl Trait for Struct {}
trait Trait {}

type Send1 = Trait + Send;
type Send2 = Trait + Send + Send;

fn main () {}

impl Trait + Send {
    fn test(&self) { println!("one"); } //~ ERROR duplicate definitions with name `test`
}

impl Trait + Send + Send {
    fn test(&self) { println!("two"); }
}
