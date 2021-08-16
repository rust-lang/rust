#![crate_name="shared"]
#![crate_type="dylib"]

// no-prefer-dynamic
// compile-flags: -C prefer-dynamic=shared,std -Z prefer-dynamic-subset

extern crate foo;

pub struct Test;

impl Test {
    pub fn new() -> Self {
        let _ = foo::foo();
        let _ = foo::bar();
        // let _ = bar::bar();
        Self
    }
}
