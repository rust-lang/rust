//@ run-pass
//! Test that users are able to retrieve generic parameters from various definitions.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2021

#![feature(rustc_private)]

extern crate rustc_middle;

extern crate rustc_driver;
extern crate rustc_interface;
#[macro_use]
extern crate rustc_public;

use rustc_public::CrateDef;
use rustc_public::ty::GenericParamDefKind;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "crate_generics";

fn test_generics() -> ControlFlow<()> {
    let local_crate = rustc_public::local_crate();

    // Check ADT generics
    let adts = local_crate.adts();
    let my_struct =
        adts.iter().find(|adt| adt.trimmed_name() == "MyStruct").expect("Failed to find MyStruct");
    let adt_generics = my_struct.generics_of();
    assert_eq!(adt_generics.params.len(), 1);
    assert_eq!(adt_generics.params[0].name.as_str(), "T");
    match &adt_generics.params[0].kind {
        GenericParamDefKind::Type { .. } => {}
        _ => panic!("Expected type parameter"),
    }

    // Check Fn generics
    let fn_defs = local_crate.fn_defs();
    let my_fn = fn_defs
        .iter()
        .find(|f| f.trimmed_name().as_str() == "my_fn")
        .expect("Failed to find my_fn");
    let fn_generics = my_fn.generics_of();
    assert_eq!(fn_generics.params.len(), 1);
    assert_eq!(fn_generics.params[0].name.as_str(), "U");

    ControlFlow::Continue(())
}

fn main() {
    let path = "generics.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "--crate-type=lib".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args, test_generics).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    std::fs::write(
        path,
        r#"
        pub struct MyStruct<T> {
            value: T,
        }

        pub fn my_fn<U>() {}
    "#
    )
}
