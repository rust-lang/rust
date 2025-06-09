/// Test for impl_stability gate.

trait Foo{}
struct Bar;

#[unstable_feature_bound(feat_foo)]
//~^ ERROR: used internally to mark impl as unstable
impl Foo for Bar{}

fn main() {
}
