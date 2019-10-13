// check that #[allow_fail] is feature-gated

#[allow_fail] //~ ERROR the `#[allow_fail]` attribute is an experimental feature
fn ok_to_fail() {
    assert!(false);
}

fn main() {}
