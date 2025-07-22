//@ run-pass
//! Test retrieval and kinds of foreign items.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2021

#![feature(rustc_private)]
#![feature(assert_matches)]

extern crate rustc_middle;

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_span;
extern crate rustc_public;

use rustc_public::{
    ty::{Abi, ForeignItemKind},
    *,
};
use std::assert_matches::assert_matches;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_foreign() -> ControlFlow<()> {
    let mods =
        local_crate().foreign_modules().into_iter().map(|def| def.module()).collect::<Vec<_>>();
    assert_eq!(mods.len(), 2);

    let rust_mod = mods.iter().find(|m| matches!(m.abi, Abi::Rust)).unwrap();
    assert_eq!(rust_mod.items().len(), 1);

    let c_mod = mods.iter().find(|m| matches!(m.abi, Abi::C { .. })).unwrap();
    let c_items = c_mod.items();
    assert_eq!(c_items.len(), 3);
    for item in c_items {
        let kind = item.kind();
        match item.name().as_str() {
            "foo" => assert_matches!(kind, ForeignItemKind::Fn(..)),
            "bar" => assert_matches!(kind, ForeignItemKind::Static(..)),
            "Baz" => assert_matches!(kind, ForeignItemKind::Type(..)),
            name => unreachable!("Unexpected item {name}"),
        };
    }
    ControlFlow::Continue(())
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `RustcPublic` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "foreign_input.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "-Cpanic=abort".to_string(),
        "--crate-type=lib".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args, || test_foreign()).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
        #![feature(extern_types)]
        #![allow(unused)]
        extern "Rust" {{
            fn rust_foo(x: i32) -> i32;
        }}
        extern "C" {{
            fn foo(x: i32) -> i32;
            static bar: i32;
            type Baz;
        }}
        "#
    )?;
    Ok(())
}
