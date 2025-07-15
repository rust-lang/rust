//@ run-pass
//! Test that users are able to use stable mir APIs to retrieve type information from a crate item
//! definition.

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

use rustc_public::ty::{Ty, ForeignItemKind};
use rustc_public::*;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "crate_def_ty";

/// Test if we can retrieve type information from different definitions.
fn test_def_tys() -> ControlFlow<()> {
    let items = rustc_public::all_local_items();
    for item in &items {
        // Type from crate items.
        let ty = item.ty();
        match item.name().as_str() {
            "STATIC_STR" => assert!(ty.kind().is_ref()),
            "CONST_U32" => assert!(ty.kind().is_integral()),
            "main" => { check_fn_def(ty) }
            _ => unreachable!("Unexpected item: `{item:?}`")
        }
    }

    let foreign_items = rustc_public::local_crate().foreign_modules();
    for item in foreign_items[0].module().items() {
        // Type from foreign items.
        let ty = item.ty();
        let item_kind = item.kind();
        let name = item.name();
        match item_kind {
            ForeignItemKind::Fn(fn_def) => {
                assert_eq!(&name, "extern_fn");
                assert_eq!(ty, fn_def.ty());
                check_fn_def(ty)
            }
            ForeignItemKind::Static(def) => {
                assert_eq!(&name, "EXT_STATIC");
                assert_eq!(ty, def.ty());
                assert!(ty.kind().is_integral())
            }
            _ => unreachable!("Unexpected kind: {item_kind:?}")
        };
    }

    ControlFlow::Continue(())
}

fn check_fn_def(ty: Ty) {
    let kind = ty.kind();
    let (def, args) = kind.fn_def().expect(&format!("Expected function type, but found: {ty}"));
    assert!(def.ty().kind().is_fn());
    assert_eq!(def.ty_with_args(args), ty);
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `RustcPublic` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "defs_ty_input.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "-Cpanic=abort".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args, test_def_tys).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
        // We would like to check intrinsic definition.
        #![feature(core_intrinsics)]
        static STATIC_STR: &str = "foo";
        const CONST_U32: u32 = 0u32;

        fn main() {{
            let _c = core::char::from_u32(99);
            let _v = Vec::<u8>::new();
            let _i = std::intrinsics::size_of::<u8>();
        }}

        extern "C" {{
            fn extern_fn(x: i32) -> i32;
            static EXT_STATIC: i32;
        }}
    "#
    )?;
    Ok(())
}
