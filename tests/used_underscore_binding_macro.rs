#![feature(plugin)]
#![plugin(clippy)]

extern crate rustc_serialize;

/// Test that we do not lint for unused underscores in a `MacroAttribute` expansion
#[deny(used_underscore_binding)]
#[derive(RustcEncodable)]
struct MacroAttributesTest {
    _foo: u32,
}

#[test]
fn macro_attributes_test() {
    let _ = MacroAttributesTest { _foo: 0 };
}
