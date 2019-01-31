#![allow(clippy::useless_attribute)] //issue #2910

#[macro_use]
extern crate serde_derive;

/// Tests that we do not lint for unused underscores in a `MacroAttribute`
/// expansion
#[deny(clippy::used_underscore_binding)]
#[derive(Deserialize)]
struct MacroAttributesTest {
    _foo: u32,
}

#[test]
fn macro_attributes_test() {
    let _ = MacroAttributesTest { _foo: 0 };
}

fn main() {}
