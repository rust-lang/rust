#![crate_name="shared"]
#![crate_type="dylib"]

// no-prefer-dynamic

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
