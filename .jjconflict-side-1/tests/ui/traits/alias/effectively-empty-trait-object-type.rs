// Test that we reject trait object types that effectively (i.e., after trait alias expansion)
// don't contain any bounds.

#![feature(trait_alias)]

trait Empty0 =;

// Nest a couple of levels deep:
trait Empty1 = Empty0;
trait Empty2 = Empty1;

// Straight list expansion:
type Type0 = dyn Empty2; //~ ERROR at least one trait is required for an object type [E0224]

// Twice:
trait Empty3 = Empty2 + Empty2;

type Type1 = dyn Empty3; //~ ERROR at least one trait is required for an object type [E0224]

fn main() {}
