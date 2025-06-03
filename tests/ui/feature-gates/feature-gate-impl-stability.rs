/// Test for impl_stability gate.

trait Foo{}
struct Bar;

#[unstable_feature_bound(feat_foo)]
//~^ ERROR: allow unstable impl
impl Foo for Bar{}

fn main() {
}
