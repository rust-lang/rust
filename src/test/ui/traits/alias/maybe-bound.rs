// build-pass (FIXME(62277): could be check-pass?)

// Test that `dyn ... + ?Sized + ...` resulting from the expansion of trait aliases is okay.

#![feature(trait_alias)]

trait Foo {}

trait S = ?Sized;

// Nest a couple of levels deep:
trait _0 = S;
trait _1 = _0;

// Straight list expansion:
type _T0 = dyn _1 + Foo;

// In second position:
type _T1 = dyn Foo + _1;

// ... and with an auto trait:
type _T2 = dyn Foo + Send + _1;

// Twice:
trait _2 = _1 + _1;

type _T3 = dyn _2 + Foo;

fn main() {}
