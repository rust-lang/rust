// run-pass

#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

// Test that you can specialize via an explicit trait hierarchy

// FIXME: this doesn't work yet...

trait Parent {}
trait Child: Parent {}

trait Foo {}

impl<T: Parent> Foo for T {}
impl<T: Child> Foo for T {}

fn main() {}
