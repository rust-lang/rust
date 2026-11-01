//@ known-bug: #149015
trait Trait1 {
    type Assoc;
}

trait Trait2 {}

impl Trait1 for <dyn Trait2 as Trait1>::Assoc {
    type Assoc = ();
}
