//@ run-pass
//! Test that users can retrieve information about inherent implementations and their self types.

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

use rustc_public::{CrateDef, CrateDefType};
use rustc_public::ty::{TyKind, RigidTy, FnDef, GenericArgKind, GenericArgs};
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "inherent_impl_test";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_inherent_impls() -> ControlFlow<()> {
    let local_crate = rustc_public::local_crate();
    let adts = local_crate.adts();

    let my_struct =
        adts.iter().find(|adt| adt.trimmed_name() == "MyStruct").expect("Failed to find MyStruct");
    let adt_generics = my_struct.generics_of();
    assert_eq!(adt_generics.params.len(), 1);
    assert_eq!(adt_generics.params[0].name.as_str(), "T");

    let impls = my_struct.inherent_impls();
    assert_eq!(impls.len(), 2, "Expected 2 inherent impls for MyStruct, found {:?}", impls);

    // We expect one generic impl `impl<T> MyStruct<T>` and one specialized impl `impl
    // MyStruct<i32>` The order might not be guaranteed, so we check both.
    let mut found_generic = false;
    let mut found_specialized = false;
    let mut generic_impl = None;
    let mut specialized_impl_self_ty = None;

    for impl_def in impls {
        let self_ty = impl_def.ty();
        let assoc_items = impl_def.associated_items();
        assert_eq!(assoc_items.len(), 2, "Expected 2 methods in impl, found {:?}", assoc_items);

        let method = assoc_items.iter().find(|m| {
            match &m.kind {
                rustc_public::ty::AssocKind::Fn { name, .. } => !name.as_str().contains("param"),
                _ => false,
            }
        }).unwrap();

        let generic_method = assoc_items.iter().find(|m| {
            match &m.kind {
                rustc_public::ty::AssocKind::Fn { name, .. } => name.as_str().contains("param"),
                _ => false,
            }
        }).unwrap();

        let method_name = match &method.kind {
            rustc_public::ty::AssocKind::Fn { name, .. } => name.as_str(),
            _ => unreachable!(),
        };

        let method_fn = FnDef(method.def_id.def_id());
        let assoc_info =
            method_fn.associated_item().expect("Expected associated item info for method");
        assert_eq!(assoc_info.def_id, method.def_id);

        let generic_method_name = match &generic_method.kind {
            rustc_public::ty::AssocKind::Fn { name, .. } => name.as_str(),
            _ => unreachable!(),
        };

        let generic_method_fn = FnDef(generic_method.def_id.def_id());
        let generic_assoc_info = generic_method_fn
            .associated_item()
            .expect("Expected associated item info for generic method");
        assert_eq!(generic_assoc_info.def_id, generic_method.def_id);

        match self_ty.kind() {
            TyKind::RigidTy(RigidTy::Adt(adt_def, args)) => {
                assert_eq!(adt_def, *my_struct);
                assert_eq!(args.0.len(), 1);

                let arg = &args.0[0];
                match arg {
                    rustc_public::ty::GenericArgKind::Type(ty) => {
                        match ty.kind() {
                            TyKind::Param(param_ty) => {
                                // This is Child<T>
                                assert_eq!(param_ty.name, "T");
                                assert_eq!(method_name, "generic_method");
                                assert_eq!(generic_method_name, "generic_method_with_param");
                                found_generic = true;
                                generic_impl = Some(impl_def.clone());

                                // Check generics of the generic impl block
                                let generics = impl_def.generics_of();
                                assert_eq!(generics.params.len(), 1);
                                assert_eq!(generics.params[0].name.as_str(), "T");

                                // Check generics of the method, which is not generic.
                                let method_generics = method_fn.generics_of();
                                assert_eq!(method_generics.params.len(), 0);

                                let generic_method_generics = generic_method_fn.generics_of();
                                assert_eq!(generic_method_generics.params.len(), 1);
                                assert_eq!(generic_method_generics.params[0].name.as_str(), "U");
                            }
                            TyKind::RigidTy(RigidTy::Int(rustc_public::ty::IntTy::I32)) => {
                                // This is Child<i32>
                                assert_eq!(method_name, "specialized_method");
                                assert_eq!(generic_method_name, "specialized_method_with_param");
                                found_specialized = true;
                                specialized_impl_self_ty = Some(self_ty.clone());

                                // Check generics of the specialized impl block (none)
                                let generics = impl_def.generics_of();
                                assert_eq!(generics.params.len(), 0);

                                // Check generics of the method
                                let method_generics = method_fn.generics_of();
                                assert_eq!(method_generics.params.len(), 0);

                                let generic_method_generics = generic_method_fn.generics_of();
                                assert_eq!(generic_method_generics.params.len(), 1);
                                assert_eq!(generic_method_generics.params[0].name.as_str(), "U");
                            }
                            _ => panic!("Unexpected generic argument type: {:?}", ty),
                        }
                    }
                    _ => panic!("Unexpected generic argument: {:?}", arg),
                }
            }
            _ => panic!("Unexpected self type: {:?}", self_ty),
        }
    }

    assert!(found_generic, "Failed to find generic impl");
    assert!(found_specialized, "Failed to find specialized impl");

    let generic_impl = generic_impl.expect("Failed to find generic impl");
    let specialized_self_ty =
        specialized_impl_self_ty.expect("Failed to find specialized impl self ty");

    let i32_ty = match specialized_self_ty.kind() {
        TyKind::RigidTy(RigidTy::Adt(_, args)) => {
            match &args.0[0] {
                rustc_public::ty::GenericArgKind::Type(t) => t.clone(),
                _ => unreachable!(),
            }
        }
        _ => unreachable!(),
    };

    let args = GenericArgs(vec![GenericArgKind::Type(i32_ty)]);
    let instantiated_ty = generic_impl.ty_with_args(&args);
    assert_eq!(instantiated_ty, specialized_self_ty);

    ControlFlow::Continue(())
}

/// This test will generate and analyze a dummy crate using the stable mir.
fn main() {
    let path = "inherent_impls.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "--crate-type=lib".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args, test_inherent_impls).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
        pub struct MyStruct<T> {{
            value: T,
        }}

        // Generic impl
        impl<T> MyStruct<T> {{
            pub fn generic_method(&self) {{}}
            pub fn generic_method_with_param<U>(&self) {{}}
        }}

        // Specialized impl
        impl MyStruct<i32> {{
            pub fn specialized_method(&self) {{}}
            pub fn specialized_method_with_param<U>(&self) {{}}
        }}
    "#
    )?;
    Ok(())
}
