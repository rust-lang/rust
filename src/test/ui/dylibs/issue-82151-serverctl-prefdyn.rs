// build-fail
// normalize-stderr-test "note: .*undefined reference to `bar::bar'" -> "note: undefined reference to `bar::bar'"
// normalize-stderr-test "note: .cc..*" -> "note: $$CC_INVOCATION"

// no-prefer-dynamic
// compile-flags: -C prefer-dynamic

// aux-build: aaa_issue_82151_bar_prefdyn.rs
// aux-build: aaa_issue_82151_foo_prefdyn.rs
// aux-build: aaa_issue_82151_shared_prefdyn.rs

extern crate shared;

fn main() {
    let _ = shared::Test::new();
}
