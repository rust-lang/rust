//@ run-pass
//! Test that types are normalized in an instance body.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2021

#![feature(rustc_private)]

extern crate rustc_middle;
#[macro_use]
extern crate rustc_smir;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate stable_mir;

use mir::mono::Instance;
use ty::{Ty, TyKind, RigidTy};
use stable_mir::*;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_stable_mir() -> ControlFlow<()> {
    let items = stable_mir::all_local_items();

    // Get all items and split generic vs monomorphic items.
    let instances: Vec<_> =
        items.into_iter().filter_map(|item| (!item.requires_monomorphization()).then(|| {
            Instance::try_from(item).unwrap()
        })).collect();
    assert_eq!(instances.len(), 1, "Expected one constant");

    for instance in instances {
        check_ty(instance.ty());
    }
    ControlFlow::Continue(())
}

fn check_ty(ty: Ty) {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Adt(def, args)) if def.kind().is_struct() => {
            // Ensure field type is also normalized
            def.variants_iter().next().unwrap().fields().into_iter().for_each(|f| {
                check_ty(f.ty_with_args(&args))
            });
        }
        TyKind::RigidTy(RigidTy::Uint(..)) => {}
        kind => unreachable!("Unexpected kind: {kind:?}")
    }
}


/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `StableMir` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "normalization_input.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "-Cpanic=abort".to_string(),
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
        pub trait Primitive {{
            type Base;
        }}

        impl Primitive for char {{
            type Base = u32;
        }}

        pub struct Wrapper<T: Primitive>(T::Base);
        pub type WrapperChar = Wrapper<char>;
        pub const NULL_CHAR: WrapperChar = Wrapper::<char>(0);
        "#
    )?;
    Ok(())
}
