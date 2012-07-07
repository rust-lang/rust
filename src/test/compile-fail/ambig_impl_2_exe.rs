// xfail-fast aux-build
// aux-build:ambig_impl_2_lib.rs
use ambig_impl_2_lib;
import ambig_impl_2_lib::methods1;
impl methods2 for uint { fn me() -> uint { self } } //~ NOTE is `methods2::me`
fn main() { 1u.me(); } //~ ERROR multiple applicable methods in scope
//~^ NOTE is `ambig_impl_2_lib::methods1::me`
