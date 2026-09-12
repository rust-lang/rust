//@ run-pass
//! Test that alias type conversion (projection/opaque types) works and returns TyKind::Alias.

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
use rustc_public::ty::{TyKind, AliasKind};
use std::ops::ControlFlow;

const CRATE_NAME: &str = "crate_alias";

fn test_alias() -> ControlFlow<()> {
    let local_crate = rustc_public::local_crate();
    let fn_defs = local_crate.fn_defs();
    let alias_fn = fn_defs
        .iter()
        .find(|f| f.trimmed_name().as_str() == "alias_fn")
        .expect("Failed to find alias_fn");
    let fn_sig = alias_fn.fn_sig().skip_binder();
    let inputs = fn_sig.inputs();
    assert_eq!(inputs.len(), 1);
    let input_ty = &inputs[0];
    match input_ty.kind() {
        TyKind::Alias(alias_kind, alias_ty) => {
            assert_eq!(alias_kind, AliasKind::Projection);
            assert_eq!(alias_ty.def_id.trimmed_name().as_str(), "MyTrait::Assoc");
        }
        _ => panic!("Expected TyKind::Alias for input type, found {:?}", input_ty),
    }

    ControlFlow::Continue(())
}

fn main() {
    let path = "alias.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "--crate-type=lib".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args, test_alias).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    std::fs::write(
        path,
        r#"
        pub trait MyTrait {
            type Assoc;
        }

        impl MyTrait for i32 {
            type Assoc = bool;
        }

        pub fn alias_fn<T: MyTrait>(val: T::Assoc) {}
    "#
    )
}
