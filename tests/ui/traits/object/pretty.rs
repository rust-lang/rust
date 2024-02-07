// Test for pretty-printing trait object types.

trait Super {
    type Assoc;
}
trait Any: Super {}
trait Fixed: Super<Assoc = u8> {}
trait FixedSub: Fixed {}

trait SuperGeneric<'a> {
    type Assoc;
}
trait AnyGeneric<'a>: SuperGeneric<'a> {}
trait FixedGeneric1<'a>: SuperGeneric<'a, Assoc = &'a u8> {}
trait FixedGeneric2<'a>: Super<Assoc = &'a u8> {}
trait FixedHrtb: for<'a> SuperGeneric<'a, Assoc = &'a u8> {}

fn dyn_super(x: &dyn Super<Assoc = u8>) { x } //~ERROR mismatched types
fn dyn_any(x: &dyn Any<Assoc = u8>) { x } //~ERROR mismatched types
fn dyn_fixed(x: &dyn Fixed) { x } //~ERROR mismatched types
fn dyn_fixed_multi(x: &dyn Fixed<Assoc = u16>) { x } //~ERROR mismatched types
fn dyn_fixed_sub(x: &dyn FixedSub) { x } //~ERROR mismatched types

fn dyn_super_generic(x: &dyn for<'a> SuperGeneric<'a, Assoc = &'a u8>) { x } //~ERROR mismatched types
fn dyn_any_generic(x: &dyn for<'a> AnyGeneric<'a, Assoc = &'a u8>) { x } //~ERROR mismatched types
fn dyn_fixed_generic1(x: &dyn for<'a> FixedGeneric1<'a>) { x } //~ERROR mismatched types
fn dyn_fixed_generic2(x: &dyn for<'a> FixedGeneric2<'a>) { x } //~ERROR mismatched types
fn dyn_fixed_generic_multi(x: &dyn for<'a> FixedGeneric1<'a, Assoc = &u8>) { x } //~ERROR mismatched types
fn dyn_fixed_hrtb(x: &dyn FixedHrtb) { x } //~ERROR mismatched types

fn main() {}
