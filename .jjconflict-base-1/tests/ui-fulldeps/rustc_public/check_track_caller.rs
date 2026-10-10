//@ run-pass
//! Test `#[track_caller]` support in rustc_public:
//! - `Instance::requires_caller_location` detects the implicit extra argument.
//! - `Body::caller_location` resolves the correct location constant.

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
use rustc_public::mir::{Body, TerminatorKind};
use rustc_public::ty::{ConstantKind, MirConst, RigidTy, TyKind};

const CRATE_NAME: &str = "input";

fn test_track_caller() -> ControlFlow<()> {
    let items = rustc_public::all_local_items();

    test_requires_caller_location(&items);
    test_caller_location_from_regular_fn(&items);
    test_caller_location_propagation(&items);

    ControlFlow::Continue(())
}

/// Check `requires_caller_location` and the extra ABI argument.
fn test_requires_caller_location(items: &[rustc_public::CrateItem]) {
    let tracked =
        items.iter().find(|item| item.name() == "input::tracked_fn").expect("missing tracked_fn");
    let not_tracked = items
        .iter()
        .find(|item| item.name() == "input::not_tracked_fn")
        .expect("missing not_tracked_fn");

    let tracked_instance = Instance::try_from(*tracked).unwrap();
    let not_tracked_instance = Instance::try_from(*not_tracked).unwrap();

    assert!(tracked_instance.requires_caller_location());
    assert!(!not_tracked_instance.requires_caller_location());

    // tracked_fn(u32) -> u32: MIR has 1 arg, ABI has 2 (extra &Location)
    let tracked_abi = tracked_instance.fn_abi().unwrap();
    assert_eq!(tracked_abi.args.len(), 2, "tracked_fn ABI should have 2 args");

    // not_tracked_fn(u32) -> u32: MIR has 1 arg, ABI has 1
    let not_tracked_abi = not_tracked_instance.fn_abi().unwrap();
    assert_eq!(not_tracked_abi.args.len(), 1, "not_tracked_fn ABI should have 1 arg");
}

/// Check that `Body::caller_location` resolves from the call site span for regular functions.
fn test_caller_location_from_regular_fn(items: &[rustc_public::CrateItem]) {
    let caller = items.iter().find(|item| item.name() == "input::caller").expect("missing caller");
    let caller_instance = Instance::try_from(*caller).unwrap();
    assert!(!caller_instance.requires_caller_location());

    let body = caller_instance.body().unwrap();
    let location = resolve_tracked_call_location(&body, None);

    // The result should be a &'static Location<'static> constant.
    assert!(
        matches!(location.ty().kind(), TyKind::RigidTy(RigidTy::Ref(..))),
        "caller_location should produce a reference type"
    );
    assert!(
        matches!(location.kind(), ConstantKind::Allocated(..)),
        "caller_location should produce an allocated constant"
    );
}

/// Check that `Body::caller_location` propagates the inherited location for `#[track_caller]` fns.
fn test_caller_location_propagation(items: &[rustc_public::CrateItem]) {
    let caller = items.iter().find(|item| item.name() == "input::caller").expect("missing caller");
    let caller_instance = Instance::try_from(*caller).unwrap();
    let caller_body = caller_instance.body().unwrap();

    let wrapper = items
        .iter()
        .find(|item| item.name() == "input::tracked_wrapper")
        .expect("missing tracked_wrapper");
    let wrapper_instance = Instance::try_from(*wrapper).unwrap();
    assert!(wrapper_instance.requires_caller_location());

    let wrapper_body = wrapper_instance.body().unwrap();

    // Use the location resolved from caller() as the inherited value.
    let inherited = resolve_tracked_call_location(&caller_body, None);

    // tracked_wrapper is #[track_caller], so pass the inherited location.
    // Without inlining, the inherited value should be returned as-is.
    let result = resolve_tracked_call_location(&wrapper_body, Some(inherited.clone()));
    assert_eq!(result, inherited, "inherited location should be propagated through");
}

/// Find the first call to a `#[track_caller]` function in the body and resolve its location.
fn resolve_tracked_call_location(body: &Body, inherited: Option<MirConst>) -> MirConst {
    for bb in &body.blocks {
        if let TerminatorKind::Call { func, .. } = &bb.terminator.kind {
            let TyKind::RigidTy(RigidTy::FnDef(def, args)) =
                func.ty(body.locals()).unwrap().kind()
            else {
                continue;
            };
            let callee = Instance::resolve(def, &args).unwrap();
            if callee.requires_caller_location() {
                return body.caller_location(&bb.terminator, inherited);
            }
        }
    }
    panic!("no call to a #[track_caller] function found in body");
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
    run!(args, test_track_caller).unwrap();
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

        #[track_caller]
        pub fn tracked_wrapper(x: u32) -> u32 {{
            tracked_fn(x)
        }}
    "#
    )?;
    Ok(())
}
