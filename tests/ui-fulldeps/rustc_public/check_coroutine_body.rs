//@ run-pass
//! Tests stable mir API for retrieving the body of a coroutine.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2024

#![feature(rustc_private)]
#![feature(assert_matches)]

extern crate rustc_middle;

extern crate rustc_driver;
extern crate rustc_interface;
#[macro_use]
extern crate rustc_public;

use std::io::Write;
use std::ops::ControlFlow;

use rustc_public::mir::Body;
use rustc_public::ty::{RigidTy, TyKind};

const CRATE_NAME: &str = "crate_coroutine_body";

fn test_coroutine_body() -> ControlFlow<()> {
    let crate_items = rustc_public::all_local_items();
    if let Some(body) = crate_items.iter().find_map(|item| {
        let item_ty = item.ty();
        if let TyKind::RigidTy(RigidTy::Coroutine(def, ..)) = &item_ty.kind() {
            if def.0.name() == "gbc::{closure#0}".to_string() {
                def.body()
            } else {
                None
            }
        } else {
            None
        }
    }) {
        check_coroutine_body(body);
    } else {
        panic!("Cannot find `gbc::{{closure#0}}`. All local items are: {:#?}", crate_items);
    }

    ControlFlow::Continue(())
}

fn check_coroutine_body(body: Body) {
    let ret_ty = &body.locals()[0].ty;
    let local_3 = &body.locals()[3].ty;
    let local_4 = &body.locals()[4].ty;

    let TyKind::RigidTy(RigidTy::Adt(def, ..)) = &ret_ty.kind()
    else {
        panic!("Expected RigidTy::Adt, got: {:#?}", ret_ty);
    };

    assert_eq!("std::task::Poll", def.0.name());

    let TyKind::RigidTy(RigidTy::Coroutine(def, ..)) = &local_3.kind()
    else {
        panic!("Expected RigidTy::Coroutine, got: {:#?}", local_3);
    };

    assert_eq!("gbc::{closure#0}::{closure#0}", def.0.name());

    let TyKind::RigidTy(RigidTy::Coroutine(def, ..)) = &local_4.kind()
    else {
        panic!("Expected RigidTy::Coroutine, got: {:#?}", local_4);
    };

    assert_eq!("gbc::{closure#0}::{closure#0}", def.0.name());
}

fn main() {
    let path = "coroutine_body.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "-Cpanic=abort".to_string(),
        "--edition".to_string(),
        "2024".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args, test_coroutine_body).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
        async fn gbc() -> i32 {{
            let a = async {{ 1 }}.await;
            a
        }}

        fn main() {{}}
    "#
    )?;
    Ok(())
}
