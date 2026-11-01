// Test for pretty-printing trait object types.

trait Super {
    type Assoc;
}
trait Any: Super {}
trait Fixed: Super<Assoc = u8> {}
trait FixedSub: Fixed {}
trait FixedStatic: Super<Assoc = &'static u8> {}

trait SuperGeneric<'a> {
    type Assoc2;
}
trait AnyGeneric<'a>: SuperGeneric<'a> {}
trait FixedGeneric1<'a>: SuperGeneric<'a, Assoc2 = &'a u8> {}
// trait FixedGeneric2<'a>: Super<Assoc = &'a u8> {} // Unsound!
trait FixedHrtb: for<'a> SuperGeneric<'a, Assoc2 = &'a u8> {}
trait AnyDifferentBinders: for<'a> SuperGeneric<'a, Assoc2 = &'a u8> + Super {}
trait FixedDifferentBinders: for<'a> SuperGeneric<'a, Assoc2 = &'a u8> + Super<Assoc = u8> {}

trait HasGat<Outer> {
    type Assoc<Inner> where Self: Sized;
}

fn dyn_super(x: &dyn Super<Assoc = u8>) { x } //~ERROR mismatched types
fn dyn_any(x: &dyn Any<Assoc = u8>) { x } //~ERROR mismatched types
fn dyn_fixed(x: &dyn Fixed) { x } //~ERROR mismatched types
fn dyn_fixed_multi(x: &dyn Fixed<Assoc = u16>) { x } //~ERROR mismatched types
fn dyn_fixed_sub(x: &dyn FixedSub) { x } //~ERROR mismatched types
fn dyn_fixed_static(x: &dyn FixedStatic) { x } //~ERROR mismatched types

fn dyn_super_generic(x: &dyn for<'a> SuperGeneric<'a, Assoc2 = &'a u8>) { x } //~ERROR mismatched types
fn dyn_any_generic(x: &dyn for<'a> AnyGeneric<'a, Assoc2 = &'a u8>) { x } //~ERROR mismatched types
fn dyn_fixed_generic1(x: &dyn for<'a> FixedGeneric1<'a>) { x } //~ERROR mismatched types
// fn dyn_fixed_generic2(x: &dyn for<'a> FixedGeneric2<'a>) { x } // Unsound!
fn dyn_fixed_generic_multi(x: &dyn for<'a> FixedGeneric1<'a, Assoc2 = &u8>) { x } //~ERROR mismatched types
fn dyn_fixed_hrtb(x: &dyn FixedHrtb) { x } //~ERROR mismatched types
fn dyn_any_different_binders(x: &dyn AnyDifferentBinders<Assoc = u8>) { x } //~ERROR mismatched types
fn dyn_fixed_different_binders(x: &dyn FixedDifferentBinders) { x } //~ERROR mismatched types

fn dyn_has_gat(x: &dyn HasGat<u8, Assoc<bool> = ()>) { x } //~ERROR mismatched types
//~^ WARN unnecessary associated type bound

fn main() {}
