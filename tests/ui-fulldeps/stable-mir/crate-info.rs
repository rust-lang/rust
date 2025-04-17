//@ run-pass
// Test that users are able to use stable mir APIs to retrieve information of the current crate

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2021

#![feature(rustc_private)]
#![feature(assert_matches)]

extern crate rustc_hir;
extern crate rustc_middle;
#[macro_use]
extern crate rustc_smir;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate stable_mir;

use rustc_hir::def::DefKind;
use stable_mir::ItemKind;
use stable_mir::crate_def::CrateDef;
use stable_mir::mir::mono::Instance;
use stable_mir::ty::{RigidTy, TyKind};
use std::assert_matches::assert_matches;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_stable_mir() -> ControlFlow<()> {
    // Get the local crate using stable_mir API.
    let local = stable_mir::local_crate();
    assert_eq!(&local.name, CRATE_NAME);

    assert_eq!(stable_mir::entry_fn(), None);

    // Find items in the local crate.
    let items = stable_mir::all_local_items();
    assert!(get_item(&items, (DefKind::Fn, "foo::bar")).is_some());

    // Find the `std` crate and assert that there is only one of it.
    assert!(stable_mir::find_crates("std").len() == 1);

    let bar = get_item(&items, (DefKind::Fn, "bar")).unwrap();
    let body = bar.expect_body();
    assert_eq!(body.locals().len(), 2);
    assert_eq!(body.blocks.len(), 1);
    let block = &body.blocks[0];
    assert_eq!(block.statements.len(), 1);
    match &block.statements[0].kind {
        stable_mir::mir::StatementKind::Assign(..) => {}
        other => panic!("{other:?}"),
    }
    match &block.terminator.kind {
        stable_mir::mir::TerminatorKind::Return => {}
        other => panic!("{other:?}"),
    }

    let foo_bar = get_item(&items, (DefKind::Fn, "foo_bar")).unwrap();
    let body = foo_bar.expect_body();
    assert_eq!(body.locals().len(), 5);
    assert_eq!(body.blocks.len(), 4);
    let block = &body.blocks[0];
    match &block.terminator.kind {
        stable_mir::mir::TerminatorKind::Call { .. } => {}
        other => panic!("{other:?}"),
    }

    let types = get_item(&items, (DefKind::Fn, "types")).unwrap();
    let body = types.expect_body();
    assert_eq!(body.locals().len(), 6);
    assert_matches!(
        body.locals()[0].ty.kind(),
        stable_mir::ty::TyKind::RigidTy(stable_mir::ty::RigidTy::Bool)
    );
    assert_matches!(
        body.locals()[1].ty.kind(),
        stable_mir::ty::TyKind::RigidTy(stable_mir::ty::RigidTy::Bool)
    );
    assert_matches!(
        body.locals()[2].ty.kind(),
        stable_mir::ty::TyKind::RigidTy(stable_mir::ty::RigidTy::Char)
    );
    assert_matches!(
        body.locals()[3].ty.kind(),
        stable_mir::ty::TyKind::RigidTy(stable_mir::ty::RigidTy::Int(stable_mir::ty::IntTy::I32))
    );
    assert_matches!(
        body.locals()[4].ty.kind(),
        stable_mir::ty::TyKind::RigidTy(stable_mir::ty::RigidTy::Uint(stable_mir::ty::UintTy::U64))
    );
    assert_matches!(
        body.locals()[5].ty.kind(),
        stable_mir::ty::TyKind::RigidTy(stable_mir::ty::RigidTy::Float(
            stable_mir::ty::FloatTy::F64
        ))
    );

    let drop = get_item(&items, (DefKind::Fn, "drop")).unwrap();
    let body = drop.expect_body();
    assert_eq!(body.blocks.len(), 2);
    let block = &body.blocks[0];
    match &block.terminator.kind {
        stable_mir::mir::TerminatorKind::Drop { .. } => {}
        other => panic!("{other:?}"),
    }

    let assert = get_item(&items, (DefKind::Fn, "assert")).unwrap();
    let body = assert.expect_body();
    assert_eq!(body.blocks.len(), 2);
    let block = &body.blocks[0];
    match &block.terminator.kind {
        stable_mir::mir::TerminatorKind::Assert { .. } => {}
        other => panic!("{other:?}"),
    }

    let monomorphic = get_item(&items, (DefKind::Fn, "monomorphic")).unwrap();
    let instance = Instance::try_from(monomorphic.clone()).unwrap();
    for block in instance.body().unwrap().blocks {
        match &block.terminator.kind {
            stable_mir::mir::TerminatorKind::Call { func, .. } => {
                let TyKind::RigidTy(ty) = func.ty(&body.locals()).unwrap().kind() else {
                    unreachable!()
                };
                let RigidTy::FnDef(def, args) = ty else { unreachable!() };
                let next_func = Instance::resolve(def, &args).unwrap();
                match next_func.body().unwrap().locals()[1].ty.kind() {
                    TyKind::RigidTy(RigidTy::Uint(_)) | TyKind::RigidTy(RigidTy::Tuple(_)) => {}
                    other => panic!("{other:?}"),
                }
            }
            stable_mir::mir::TerminatorKind::Return => {}
            other => panic!("{other:?}"),
        }
    }

    let foo_const = get_item(&items, (DefKind::Const, "FOO")).unwrap();
    // Ensure we don't panic trying to get the body of a constant.
    foo_const.expect_body();

    let locals_fn = get_item(&items, (DefKind::Fn, "locals")).unwrap();
    let body = locals_fn.expect_body();
    assert_eq!(body.locals().len(), 4);
    assert_matches!(
        body.ret_local().ty.kind(),
        stable_mir::ty::TyKind::RigidTy(stable_mir::ty::RigidTy::Char)
    );
    assert_eq!(body.arg_locals().len(), 2);
    assert_matches!(
        body.arg_locals()[0].ty.kind(),
        stable_mir::ty::TyKind::RigidTy(stable_mir::ty::RigidTy::Int(stable_mir::ty::IntTy::I32))
    );
    assert_matches!(
        body.arg_locals()[1].ty.kind(),
        stable_mir::ty::TyKind::RigidTy(stable_mir::ty::RigidTy::Uint(stable_mir::ty::UintTy::U64))
    );
    assert_eq!(body.inner_locals().len(), 1);
    // If conditions have an extra inner local to hold their results
    assert_matches!(
        body.inner_locals()[0].ty.kind(),
        stable_mir::ty::TyKind::RigidTy(stable_mir::ty::RigidTy::Bool)
    );

    ControlFlow::Continue(())
}

// Use internal API to find a function in a crate.
fn get_item<'a>(
    items: &'a stable_mir::CrateItems,
    item: (DefKind, &str),
) -> Option<&'a stable_mir::CrateItem> {
    items.iter().find(|crate_item| {
        matches!(
            (item.0, crate_item.kind()),
            (DefKind::Fn, ItemKind::Fn) | (DefKind::Const, ItemKind::Const)
        ) && crate_item.name() == item.1
    })
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `StableMir` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "input.rs";
    generate_input(&path).unwrap();
    let args = vec![
        "rustc".to_string(),
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
    pub const FOO: u32 = 1 + 2;

    fn generic<T, const U: usize>(t: T) -> [(); U] {{
        _ = t;
        [(); U]
    }}

    pub fn monomorphic() {{
        generic::<(), 5>(());
        generic::<u32, 0>(45);
    }}

    mod foo {{
        pub fn bar(i: i32) -> i64 {{
            i as i64
        }}
    }}

    pub fn bar(x: i32) -> i32 {{
        x
    }}

    pub fn foo_bar(x: i32, y: i32) -> i64 {{
        let x_64 = foo::bar(x);
        let y_64 = foo::bar(y);
        x_64.wrapping_add(y_64)
    }}

    pub fn types(b: bool, _: char, _: i32, _: u64, _: f64) -> bool {{
        b
    }}

    pub fn drop(_: String) {{}}

    pub fn assert(x: i32) -> i32 {{
        x + 1
    }}

    pub fn locals(a: i32, _: u64) -> char {{
        if a > 5 {{
            'a'
        }} else {{
            'b'
        }}
    }}"#
    )?;
    Ok(())
}
