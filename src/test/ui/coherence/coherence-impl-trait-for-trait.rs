// Test that we give suitable error messages when the user attempts to
// impl a trait `Trait` for its own object type.

// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

trait Foo { fn dummy(&self) { } }
trait Bar: Foo { }
trait Baz: Bar { }

// Supertraits of Baz are not legal:
impl Foo for Baz { }
//[old]~^ ERROR E0371
//[re]~^^ ERROR E0371
impl Bar for Baz { }
//[old]~^ ERROR E0371
//[re]~^^ ERROR E0371
impl Baz for Baz { }
//[old]~^ ERROR E0371
//[re]~^^ ERROR E0371

// But other random traits are:
trait Other { }
impl Other for Baz { } // OK, Other not a supertrait of Baz

fn main() { }
