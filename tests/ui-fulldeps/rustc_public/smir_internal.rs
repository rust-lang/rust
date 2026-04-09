//@ run-pass
//! Test that users are able to use retrieve internal constructs from stable ones to help with
//! the migration.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2021

#![feature(rustc_private)]
#![feature(assert_matches)]

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_middle;
#[macro_use]
extern crate rustc_public;

use rustc_middle::ty::TyCtxt;
use rustc_public::rustc_internal;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

fn test_translation(tcx: TyCtxt<'_>) -> ControlFlow<()> {
    let main_fn = rustc_public::entry_fn().unwrap();
    let body = main_fn.expect_body();
    let orig_ty = body.locals()[0].ty;
    let rustc_ty = rustc_internal::internal(tcx, &orig_ty);
    assert!(rustc_ty.is_unit());
    ControlFlow::Continue(())
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `RustcPublic` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "internal_input.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run_with_tcx!(args, test_translation).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
    pub fn main() {{
    }}
    "#
    )?;
    Ok(())
}
