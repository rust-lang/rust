//@ run-pass
//! Test that users are able to use retrieve internal constructs from stable ones to help with
//! the migration.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ ignore-windows-gnu mingw has troubles with linking https://github.com/rust-lang/rust/pull/116837
//@ edition: 2021

#![feature(rustc_private)]
#![feature(assert_matches)]

#[macro_use]
extern crate rustc_smir;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate stable_mir;

use rustc_middle::ty::TyCtxt;
use rustc_smir::rustc_internal;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

fn test_translation(tcx: TyCtxt<'_>) -> ControlFlow<()> {
    let main_fn = stable_mir::entry_fn().unwrap();
    let body = main_fn.body();
    let orig_ty = body.locals()[0].ty;
    let rustc_ty = rustc_internal::internal(tcx, &orig_ty);
    assert!(rustc_ty.is_unit());
    ControlFlow::Continue(())
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `StableMir` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "internal_input.rs";
    generate_input(&path).unwrap();
    let args = vec![
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
