//@ run-pass
//! Tests stable mir API for retrieving the body of a closure.

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

use std::io::Write;
use std::ops::ControlFlow;

// use rustc_public::mir::{Body, ConstOperand, Operand, TerminatorKind};
use rustc_public::ty::{ClosureArgs, RigidTy, TyKind};
use rustc_public::crate_def::DefId;

const CRATE_NAME: &str = "crate_closure_body";

fn test_closure_args() -> ControlFlow<()> {
    let crate_items = rustc_public::all_local_items();
    for item in crate_items {
        let item_ty = item.ty();
        match &item_ty.kind() {
            TyKind::RigidTy(RigidTy::Closure(closure_def, args)) => {
                assert!(DefId::is_closure_like(&closure_def.0));
                let closure_args = args.as_closure();
                check_closure_args(closure_args);
            }
            _ => {}
        }
    }

    ControlFlow::Continue(())
}
fn check_closure_args(args: ClosureArgs) {
    let parts = args.split();

    // split(): last 3 args are the synthetics, rest are parent args.
    assert!(
        parts.parent_args.len() + 3 == args.args.0.len(),
        "split should partition args into parent + 3 synthetics",
    );

    // kind_ty(): must be a type arg, and must match the split field.
    assert!(
        args.kind_ty() == *parts.closure_kind_ty
            && matches!(args.kind_ty().kind(), TyKind::RigidTy(_)),
        "kind_ty mismatch or expected a rigid type, got: {:?}",
        args.kind_ty()
    );

    // sig_as_fn_ptr_ty(): must be a fn-pointer type and match the split field.
    assert!(
        args.sig_as_fn_ptr_ty() == *parts.closure_sig_as_fn_ptr_ty
            && matches!(
                args.sig_as_fn_ptr_ty().kind(),
                TyKind::RigidTy(RigidTy::FnPtr(_))
            ),
        "sig_as_fn_ptr_ty mismatch or expected a FnPtr, got: {:?}",
        args.sig_as_fn_ptr_ty().kind()
    );

    // tupled_upvars_ty(): must be a tuple type and match the split field.
    assert!(
        args.tupled_upvars_ty() == *parts.tupled_upvars_ty
            && matches!(
                args.tupled_upvars_ty().kind(),
                TyKind::RigidTy(RigidTy::Tuple(_))
            ),
        "tupled_upvars_ty mismatch or expected a Tuple, got: {:?}",
        args.tupled_upvars_ty().kind()
    );
}



fn main() {
    let path = "closure_body.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "-Cpanic=abort".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args, test_closure_args).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
        fn incr(y: i32) -> i32 {{
            y + 1
        }}

        fn main() {{
            let cl_incr = |x: i32| {{
                incr(x)
            }};

            let _= cl_incr(5);
        }}
    "#
    )?;
    Ok(())
}
