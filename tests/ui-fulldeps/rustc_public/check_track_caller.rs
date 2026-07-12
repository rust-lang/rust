//@ run-pass
//! Test that users can query `Instance::requires_caller_location` to detect
//! `#[track_caller]` functions and the implicit extra argument in their ABI.

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

use std::io::Write;
use std::ops::ControlFlow;

use rustc_public::crate_def::CrateDef;
use rustc_public::mir::mono::Instance;
use rustc_public::mir::TerminatorKind;
use rustc_public::ty::{RigidTy, TyKind};

const CRATE_NAME: &str = "input";

fn test_stable_mir() -> ControlFlow<()> {
    let items = rustc_public::all_local_items();

    let tracked = items
        .iter()
        .find(|item| item.name() == "input::tracked_fn")
        .expect("missing tracked_fn");
    let not_tracked = items
        .iter()
        .find(|item| item.name() == "input::not_tracked_fn")
        .expect("missing not_tracked_fn");
    let caller =
        items.iter().find(|item| item.name() == "input::caller").expect("missing caller");

    let tracked_instance = Instance::try_from(*tracked).unwrap();
    let not_tracked_instance = Instance::try_from(*not_tracked).unwrap();

    // Check requires_caller_location
    assert!(
        tracked_instance.requires_caller_location(),
        "tracked_fn should require caller location"
    );
    assert!(
        !not_tracked_instance.requires_caller_location(),
        "not_tracked_fn should NOT require caller location"
    );

    // Verify that the ABI reflects the extra argument.
    let tracked_abi = tracked_instance.fn_abi().unwrap();
    let not_tracked_abi = not_tracked_instance.fn_abi().unwrap();

    // tracked_fn(u32) -> u32: MIR has 1 arg, ABI has 2 (extra &Location)
    assert_eq!(tracked_abi.args.len(), 2, "tracked_fn ABI should have 2 args");
    // not_tracked_fn(u32) -> u32: MIR has 1 arg, ABI has 1
    assert_eq!(not_tracked_abi.args.len(), 1, "not_tracked_fn ABI should have 1 arg");

    // Check that calling a #[track_caller] function from the caller's body
    // resolves to an instance that requires caller location.
    let caller_instance = Instance::try_from(*caller).unwrap();
    let caller_body = caller_instance.body().unwrap();
    for bb in &caller_body.blocks {
        if let TerminatorKind::Call { func, .. } = &bb.terminator.kind {
            let TyKind::RigidTy(RigidTy::FnDef(def, args)) =
                func.ty(caller_body.locals()).unwrap().kind()
            else {
                continue;
            };
            let callee = Instance::resolve(def, &args).unwrap();
            if callee.mangled_name().contains("tracked_fn") {
                assert!(
                    callee.requires_caller_location(),
                    "resolved callee should require caller location"
                );
            }
        }
    }

    ControlFlow::Continue(())
}

fn main() {
    let path = "track_caller_input.rs";
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
        #[track_caller]
        pub fn tracked_fn(x: u32) -> u32 {{
            x + 1
        }}

        pub fn not_tracked_fn(x: u32) -> u32 {{
            x + 1
        }}

        pub fn caller() -> u32 {{
            tracked_fn(42)
        }}
    "#
    )?;
    Ok(())
}
