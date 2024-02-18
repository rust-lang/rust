//@ run-pass
//@ only-x86_64
//@ aux-build:using-target-feature-unstable.rs

extern crate using_target_feature_unstable;

fn main() {
    unsafe {
        using_target_feature_unstable::foo();
    }
}
