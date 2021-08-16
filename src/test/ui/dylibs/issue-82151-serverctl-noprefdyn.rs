// build-fail

// no-prefer-dynamic

// aux-build: aaa_issue_82151_bar_noprefdyn.rs
// aux-build: aaa_issue_82151_foo_noprefdyn.rs
// aux-build: aaa_issue_82151_shared_noprefdyn.rs

extern crate shared;

fn main() {
    let _ = shared::Test::new();
}
