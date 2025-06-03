/// Test for impl_stability gate.

trait Foo{}
struct Bar;

// TODO: make the error message below nicer
#[unstable_feature_bound(feat_foo)]
//~^ ERROR: allow unstable impl
impl Foo for Bar{}

fn main() {
}
