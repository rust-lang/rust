//@ run-pass
//! Test that item kind works as expected.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2021

#![feature(rustc_private)]
#![feature(assert_matches)]

extern crate rustc_middle;

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_public;

use rustc_public::*;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_item_kind() -> ControlFlow<()> {
    let items = rustc_public::all_local_items();
    assert_eq!(items.len(), 4);
    // Constructor item.
    for item in items {
        let expected_kind = match item.name().as_str() {
            "Dummy" => ItemKind::Ctor(CtorKind::Fn),
            "dummy" => ItemKind::Fn,
            "unit" => ItemKind::Fn,
            "DUMMY_CONST" => ItemKind::Const,
            name => unreachable!("Unexpected item {name}"),
        };
        assert_eq!(item.kind(), expected_kind, "Mismatched type for {}", item.name());
    }
    ControlFlow::Continue(())
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `RustcPublic` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "item_kind_input.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "-Cpanic=abort".to_string(),
        "--crate-type=lib".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args, test_item_kind).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
        pub struct Dummy(u32);
        pub const DUMMY_CONST: Dummy = Dummy(0);
        pub struct DummyUnit;

        pub fn dummy() -> Dummy {{
            Dummy(5)
        }}

        pub fn unit() -> DummyUnit {{
            DummyUnit
        }}
        "#
    )?;
    Ok(())
}
