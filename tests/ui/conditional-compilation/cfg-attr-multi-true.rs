// Test that cfg_attr with multiple attributes actually emits both attributes.
// This is done by emitting two attributes that cause new warnings, and then
// triggering those warnings.

//@ check-pass

#![warn(unused_must_use)]

#[cfg_attr(true, deprecated, must_use)]
struct MustUseDeprecated {}

impl MustUseDeprecated { //~ warning: use of deprecated
    fn new() -> MustUseDeprecated { //~ warning: use of deprecated
        MustUseDeprecated {} //~ warning: use of deprecated
    }
}

fn main() {
    MustUseDeprecated::new(); //~ warning: use of deprecated
    //~| warning: unused `MustUseDeprecated` that must be used
}
