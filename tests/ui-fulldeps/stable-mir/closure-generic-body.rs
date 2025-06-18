//@ run-pass
//! Tests stable mir API for retrieving the body of a closure.

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

use std::io::Write;
use std::ops::ControlFlow;

use stable_mir::mir::{Body, ConstOperand, Operand, TerminatorKind};
use stable_mir::ty::{FnDef, RigidTy, TyKind};

const CRATE_NAME: &str = "crate_closure_body";

fn test_closure_body() -> ControlFlow<()> {
    let crate_items = stable_mir::all_local_items();
    for item in crate_items {
        let item_ty = item.ty();
        match &item_ty.kind() {
            TyKind::RigidTy(RigidTy::Closure(closure_def, _)) => {
                let closure_body = closure_def.body().unwrap();
                check_incr_closure_body(closure_body);
            }
            _ => {}
        }
    }

    ControlFlow::Continue(())
}

fn check_incr_closure_body(body: Body) {
    let first_block = &body.blocks[0];
    let TerminatorKind::Call { func: Operand::Constant(ConstOperand { const_, .. }), .. } =
        &first_block.terminator.kind
    else {
        panic!("expected Call Terminator, got: ");
    };

    let TyKind::RigidTy(RigidTy::FnDef(FnDef(def_id), ..), ..) = const_.ty().kind() else {
        panic!("expected FnDef");
    };

    assert_eq!(def_id.name(), "id");
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
    run!(args, test_closure_body).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
        fn id<T>(y: T) -> T {{
            y
        }}

        fn main() {{
            let cl_id= |x| {{
                id(x)
            }};

            let _= cl_id(5);
        }}
    "#
    )?;
    Ok(())
}
