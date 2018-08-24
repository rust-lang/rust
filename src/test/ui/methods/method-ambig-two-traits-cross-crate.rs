// Test an ambiguity scenario where one copy of the method is available
// from a trait imported from another crate.

// aux-build:ambig_impl_2_lib.rs
extern crate ambig_impl_2_lib;
use ambig_impl_2_lib::me;
trait me2 {
    fn me(&self) -> usize;
}
impl me2 for usize { fn me(&self) -> usize { *self } }
fn main() { 1_usize.me(); } //~ ERROR E0034
