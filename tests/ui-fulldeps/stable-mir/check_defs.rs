//@ run-pass
//! Test that users are able to use stable mir APIs to retrieve information about crate definitions.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2021

#![feature(rustc_private)]
#![feature(assert_matches)]

extern crate rustc_middle;
#[macro_use]
extern crate rustc_smir;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate stable_mir;

use std::assert_matches::assert_matches;
use mir::{mono::Instance, TerminatorKind::*};
use stable_mir::mir::mono::InstanceKind;
use stable_mir::ty::{RigidTy, TyKind, Ty, UintTy};
use stable_mir::*;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_stable_mir() -> ControlFlow<()> {
    let entry = stable_mir::entry_fn().unwrap();
    let main_fn = Instance::try_from(entry).unwrap();
    assert_eq!(main_fn.name(), "main");
    assert_eq!(main_fn.trimmed_name(), "main");

    let instances = get_instances(main_fn.body().unwrap());
    assert_eq!(instances.len(), 3);
    test_fn(instances[0], "from_u32", "std::char::from_u32", "core");
    test_fn(instances[1], "Vec::<u8>::new", "std::vec::Vec::<u8>::new", "alloc");
    test_fn(instances[2], "ctpop::<u8>", "std::intrinsics::ctpop::<u8>", "core");
    test_vec_new(instances[1]);
    ControlFlow::Continue(())
}

fn test_fn(instance: Instance, expected_trimmed: &str, expected_qualified: &str, krate: &str) {
    let trimmed = instance.trimmed_name();
    let qualified = instance.name();
    assert_eq!(&trimmed, expected_trimmed);
    assert_eq!(&qualified, expected_qualified);

    if instance.kind == InstanceKind::Intrinsic {
        let intrinsic = instance.intrinsic_name().unwrap();
        let (trimmed_base, _trimmed_args) = trimmed.split_once("::").unwrap();
        assert_eq!(intrinsic, trimmed_base);
        return;
    }
    assert!(instance.intrinsic_name().is_none());

    let item = CrateItem::try_from(instance).unwrap();
    let trimmed = item.trimmed_name();
    let qualified = item.name();
    assert_eq!(trimmed, expected_trimmed.replace("u8", "T"));
    assert_eq!(qualified, expected_qualified.replace("u8", "T"));
    assert_eq!(&item.krate().name, krate);
}

fn extract_elem_ty(ty: Ty) -> Ty {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Adt(_, args)) => {
            *args.0[0].expect_ty()
        }
        _ => unreachable!("Expected Vec ADT, but found: {ty:?}")
    }
}

/// Check signature and type of `Vec::<u8>::new` and its generic version.
fn test_vec_new(instance: mir::mono::Instance) {
    let sig = instance.fn_abi().unwrap();
    assert_eq!(&sig.args, &[]);
    let elem_ty = extract_elem_ty(sig.ret.ty);
    assert_matches!(elem_ty.kind(), TyKind::RigidTy(RigidTy::Uint(UintTy::U8)));

    // Get the signature for Vec::<T>::new.
    let item = CrateItem::try_from(instance).unwrap();
    let ty = item.ty();
    let gen_sig = ty.kind().fn_sig().unwrap().skip_binder();
    let gen_ty = extract_elem_ty(gen_sig.output());
    assert_matches!(gen_ty.kind(), TyKind::Param(_));
}

/// Inspect the instance body
fn get_instances(body: mir::Body) -> Vec<Instance> {
    body.blocks.iter().filter_map(|bb| {
        match &bb.terminator.kind {
            Call { func, .. } => {
                let TyKind::RigidTy(ty) = func.ty(body.locals()).unwrap().kind() else { unreachable!
                () };
                let RigidTy::FnDef(def, args) = ty else { unreachable!() };
                Instance::resolve(def, &args).ok()
            }
            _ => {
                None
            }
        }
    }).collect::<Vec<_>>()
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `StableMir` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "defs_input.rs";
    generate_input(&path).unwrap();
    let args = vec![
        "rustc".to_string(),
        "-Cpanic=abort".to_string(),
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
        #![feature(core_intrinsics)]

        fn main() {{
            let _c = core::char::from_u32(99);
            let _v = Vec::<u8>::new();
            let _i = std::intrinsics::ctpop::<u8>(0);
        }}
    "#
    )?;
    Ok(())
}
