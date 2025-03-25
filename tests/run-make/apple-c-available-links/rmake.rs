//! Test that using `__builtin_available` in C (`@available` in Objective-C)
//! successfully links (because `std` provides the required symbols).

//@ only-apple __builtin_available is (mostly) specific to Apple platforms.

use run_make_support::{cargo, path, target};

fn main() {
    let target_dir = path("target");
    cargo().args(&["build", "--target", &target()]).env("CARGO_TARGET_DIR", &target_dir).run();
}
