//@ run-pass
//! Test information regarding intrinsics and ensure we can retrieve the fallback body if it exists.
//!
//! This tests relies on the intrinsics implementation, and requires one intrinsic with and one
//! without a body. It doesn't matter which intrinsic is called here, and feel free to update that
//! if needed.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote

#![feature(rustc_private)]
#![feature(assert_matches)]

extern crate rustc_middle;
extern crate rustc_hir;

extern crate rustc_driver;
extern crate rustc_interface;
#[macro_use]
extern crate rustc_public;

use rustc_public::mir::mono::{Instance, InstanceKind};
use rustc_public::mir::visit::{Location, MirVisitor};
use rustc_public::mir::{LocalDecl, Terminator, TerminatorKind};
use rustc_public::ty::{FnDef, GenericArgs, RigidTy, TyKind};
use std::assert_matches::assert_matches;
use std::convert::TryFrom;
use std::io::Write;
use std::ops::ControlFlow;

/// This function tests that we can correctly get type information from binary operations.
fn test_intrinsics() -> ControlFlow<()> {
    // Find items in the local crate.
    let main_def = rustc_public::all_local_items()[0];
    let main_instance = Instance::try_from(main_def).unwrap();
    let main_body = main_instance.body().unwrap();
    let mut visitor = CallsVisitor { locals: main_body.locals(), calls: Default::default() };
    visitor.visit_body(&main_body);

    let calls = visitor.calls;
    assert_eq!(calls.len(), 3, "Expected 3 calls, but found: {calls:?}");
    for (fn_def, args) in calls.into_iter() {
        check_instance(&Instance::resolve(fn_def, &args).unwrap());
        check_def(fn_def);
    }

    ControlFlow::Continue(())
}

/// This check is unfortunately tight to the implementation of intrinsics.
///
/// We want to ensure that StableMIR can handle intrinsics with and without fallback body:
/// for intrinsics without a body, obviously we cannot expose anything.
///
/// If by any chance this test breaks because you changed how an intrinsic is implemented, please
/// update the test to invoke a different intrinsic.
fn check_instance(instance: &Instance) {
    assert_eq!(instance.kind, InstanceKind::Intrinsic);
    let name = instance.intrinsic_name().unwrap();
    if instance.has_body() {
        let Some(body) = instance.body() else { unreachable!("Expected a body") };
        assert!(!body.blocks.is_empty());
        assert_eq!(&name, "select_unpredictable");
    } else {
        assert!(instance.body().is_none());
        assert_matches!(name.as_str(), "size_of_val" | "vtable_size");
    }
}

fn check_def(fn_def: FnDef) {
    assert!(fn_def.is_intrinsic());
    let intrinsic = fn_def.as_intrinsic().unwrap();
    assert_eq!(fn_def, intrinsic.into());

    let name = intrinsic.fn_name();
    match name.as_str() {
        "select_unpredictable" => {
            assert!(!intrinsic.must_be_overridden());
            assert!(fn_def.has_body());
        }
        "vtable_size" | "size_of_val" => {
            assert!(intrinsic.must_be_overridden());
            assert!(!fn_def.has_body());
        }
        _ => unreachable!("Unexpected intrinsic: {}", name),
    }
}

struct CallsVisitor<'a> {
    locals: &'a [LocalDecl],
    calls: Vec<(FnDef, GenericArgs)>,
}

impl<'a> MirVisitor for CallsVisitor<'a> {
    fn visit_terminator(&mut self, term: &Terminator, _loc: Location) {
        match &term.kind {
            TerminatorKind::Call { func, .. } => {
                let TyKind::RigidTy(RigidTy::FnDef(def, args)) =
                    func.ty(self.locals).unwrap().kind()
                    else {
                        return;
                    };
                self.calls.push((def, args.clone()));
            }
            _ => {}
        }
    }
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `RustcPublic` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "binop_input.rs";
    generate_input(&path).unwrap();
    let args = &["rustc".to_string(), "--crate-type=lib".to_string(), path.to_string()];
    run!(args, test_intrinsics).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
        #![feature(core_intrinsics)]
        use std::intrinsics::*;
        pub fn use_intrinsics(init: bool) -> bool {{
            let vtable_sz = unsafe {{ vtable_size(0 as *const ()) }};
            let sz = unsafe {{ size_of_val("hi") }};
            select_unpredictable(init && sz == 2, false, true)
        }}
        "#
    )?;
    Ok(())
}
