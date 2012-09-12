// xfail-fast aux-build
// aux-build:ambig_impl_2_lib.rs
extern mod ambig_impl_2_lib;
use ambig_impl_2_lib::me;
trait me {
    fn me() -> uint;
}
impl uint: me { fn me() -> uint { self } } //~ NOTE is `__extensions__::me`
fn main() { 1u.me(); } //~ ERROR multiple applicable methods in scope
//~^ NOTE is `ambig_impl_2_lib::__extensions__::me`
