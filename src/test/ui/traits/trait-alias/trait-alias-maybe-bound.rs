// Test that `dyn ... + ?Sized + ...` resulting from the expansion of trait aliases is okay.

#![feature(trait_alias)]

trait S = ?Sized;

// Nest a couple of levels deep:
trait _0 = S;
trait _1 = _0;

// Straight list expansion:
type _T0 = dyn _1;
//~^ ERROR at least one non-builtin trait is required for an object type [E0224]

// In second position:
type _T1 = dyn Copy + _1;

// ... and with an auto trait:
type _T2 = dyn Copy + Send + _1;

// Twice:
trait _2 = _1 + _1;

type _T3 = dyn _2;
//~^ ERROR at least one non-builtin trait is required for an object type [E0224]

fn main() {}
