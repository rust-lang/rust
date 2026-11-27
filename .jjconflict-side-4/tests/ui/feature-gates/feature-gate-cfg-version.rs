//! Feature gate test for `cfg_version`.
//!
//! Tracking issue: #64796.

#[cfg(version("1.42"))]
//~^ ERROR `cfg(version)` is experimental and subject to change
fn bar() {}

fn main() {
    assert!(cfg!(version("1.42")));
    //~^ ERROR `cfg(version)` is experimental and subject to change
}
