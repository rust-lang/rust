// check that #[allow_fail] is feature-gated

#[allow_fail] //~ ERROR allow_fail attribute is currently unstable
fn ok_to_fail() {
    assert!(false);
}

fn main() {}
