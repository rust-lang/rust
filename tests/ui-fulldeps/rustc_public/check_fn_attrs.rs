//@ run-pass
//! Test that users are able to query function-level constness and asyncness.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2021

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_middle;
#[macro_use]
extern crate rustc_public;

use rustc_public::crate_def::CrateDef;
use rustc_public::ty::{Asyncness, Constness, FnDef};
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

fn test_stable_mir() -> ControlFlow<()> {
    let fns = rustc_public::local_crate().fn_defs();

    check_fn(&fns, "input::const_sync", Constness::Const, Asyncness::NotAsync);
    check_fn(&fns, "input::async_fn", Constness::NotConst, Asyncness::Async);
    check_fn(&fns, "input::plain", Constness::NotConst, Asyncness::NotAsync);
    check_fn(&fns, "input::Widget::assoc_const", Constness::Const, Asyncness::NotAsync);
    check_fn(&fns, "input::Widget::assoc_async", Constness::NotConst, Asyncness::Async);
    check_fn(&fns, "input::Widget::assoc_plain", Constness::NotConst, Asyncness::NotAsync);

    ControlFlow::Continue(())
}

fn check_fn(fns: &[FnDef], name: &str, constness: Constness, asyncness: Asyncness) {
    let fn_def =
        fns.iter().find(|def| def.name() == name).unwrap_or_else(|| panic!("missing {name}"));
    assert_eq!(fn_def.constness(), constness, "wrong constness for {}", fn_def.name());
    assert_eq!(fn_def.asyncness(), asyncness, "wrong asyncness for {}", fn_def.name());
}

fn main() {
    let path = "fn_attrs_input.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "--edition=2021".to_string(),
        "--crate-type=lib".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args, test_stable_mir).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
        pub const fn const_sync() -> u32 {{
            1
        }}

        pub async fn async_fn() -> u32 {{
            2
        }}

        pub fn plain() -> u32 {{
            3
        }}

        pub struct Widget;

        impl Widget {{
            pub const fn assoc_const() -> u32 {{
                4
            }}

            pub async fn assoc_async() -> u32 {{
                5
            }}

            pub fn assoc_plain() -> u32 {{
                6
            }}
        }}
    "#
    )?;
    Ok(())
}
