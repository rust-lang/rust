// run-pass
// Test that users are able to use stable mir APIs to retrieve information of the current crate

// ignore-stage1
// ignore-cross-compile
// ignore-remote
// ignore-windows-gnu mingw has troubles with linking https://github.com/rust-lang/rust/pull/116837
// edition: 2021

#![feature(rustc_private)]
#![feature(assert_matches)]
#![feature(control_flow_enum)]

extern crate rustc_hir;
extern crate rustc_middle;
#[macro_use]
extern crate rustc_smir;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate stable_mir;

use rustc_hir::def::DefKind;
use rustc_middle::ty::TyCtxt;
use rustc_smir::rustc_internal;

use stable_mir::fold::Foldable;
use std::assert_matches::assert_matches;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_stable_mir(_tcx: TyCtxt<'_>) -> ControlFlow<()> {
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
    let body = bar.body();
    assert_eq!(body.locals.len(), 2);
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
    let body = foo_bar.body();
    assert_eq!(body.locals.len(), 5);
    assert_eq!(body.blocks.len(), 4);
    let block = &body.blocks[0];
    match &block.terminator.kind {
        stable_mir::mir::TerminatorKind::Call { .. } => {}
        other => panic!("{other:?}"),
    }

    let types = get_item(&items, (DefKind::Fn, "types")).unwrap();
    let body = types.body();
    assert_eq!(body.locals.len(), 6);
    assert_matches!(
        body.locals[0].ty.kind(),
        stable_mir::ty::TyKind::RigidTy(stable_mir::ty::RigidTy::Bool)
    );
    assert_matches!(
        body.locals[1].ty.kind(),
        stable_mir::ty::TyKind::RigidTy(stable_mir::ty::RigidTy::Bool)
    );
    assert_matches!(
        body.locals[2].ty.kind(),
        stable_mir::ty::TyKind::RigidTy(stable_mir::ty::RigidTy::Char)
    );
    assert_matches!(
        body.locals[3].ty.kind(),
        stable_mir::ty::TyKind::RigidTy(stable_mir::ty::RigidTy::Int(stable_mir::ty::IntTy::I32))
    );
    assert_matches!(
        body.locals[4].ty.kind(),
        stable_mir::ty::TyKind::RigidTy(stable_mir::ty::RigidTy::Uint(stable_mir::ty::UintTy::U64))
    );
    assert_matches!(
        body.locals[5].ty.kind(),
        stable_mir::ty::TyKind::RigidTy(stable_mir::ty::RigidTy::Float(
            stable_mir::ty::FloatTy::F64
        ))
    );

    let drop = get_item(&items, (DefKind::Fn, "drop")).unwrap();
    let body = drop.body();
    assert_eq!(body.blocks.len(), 2);
    let block = &body.blocks[0];
    match &block.terminator.kind {
        stable_mir::mir::TerminatorKind::Drop { .. } => {}
        other => panic!("{other:?}"),
    }

    let assert = get_item(&items, (DefKind::Fn, "assert")).unwrap();
    let body = assert.body();
    assert_eq!(body.blocks.len(), 2);
    let block = &body.blocks[0];
    match &block.terminator.kind {
        stable_mir::mir::TerminatorKind::Assert { .. } => {}
        other => panic!("{other:?}"),
    }

    let monomorphic = get_item(&items, (DefKind::Fn, "monomorphic")).unwrap();
    for block in monomorphic.body().blocks {
        match &block.terminator.kind {
            stable_mir::mir::TerminatorKind::Call { func, .. } => match func {
                stable_mir::mir::Operand::Constant(c) => match &c.literal.literal {
                    stable_mir::ty::ConstantKind::Allocated(alloc) => {
                        assert!(alloc.bytes.is_empty());
                        match c.literal.ty.kind() {
                            stable_mir::ty::TyKind::RigidTy(stable_mir::ty::RigidTy::FnDef(
                                def,
                                mut args,
                            )) => {
                                let func = def.body();
                                match func.locals[1].ty
                                    .fold(&mut args)
                                    .continue_value()
                                    .unwrap()
                                    .kind()
                                {
                                    stable_mir::ty::TyKind::RigidTy(
                                        stable_mir::ty::RigidTy::Uint(_),
                                    ) => {}
                                    stable_mir::ty::TyKind::RigidTy(
                                        stable_mir::ty::RigidTy::Tuple(_),
                                    ) => {}
                                    other => panic!("{other:?}"),
                                }
                            }
                            other => panic!("{other:?}"),
                        }
                    }
                    other => panic!("{other:?}"),
                },
                other => panic!("{other:?}"),
            },
            stable_mir::mir::TerminatorKind::Return => {}
            other => panic!("{other:?}"),
        }
    }

    let foo_const = get_item(&items, (DefKind::Const, "FOO")).unwrap();
    // Ensure we don't panic trying to get the body of a constant.
    foo_const.body();

    ControlFlow::Continue(())
}

// Use internal API to find a function in a crate.
fn get_item<'a>(
    items: &'a stable_mir::CrateItems,
    item: (DefKind, &str),
) -> Option<&'a stable_mir::CrateItem> {
    items.iter().find(|crate_item| {
        crate_item.kind().to_string() == format!("{:?}", item.0) && crate_item.name() == item.1
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
    run!(args, tcx, test_stable_mir(tcx)).unwrap();
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
    }}"#
    )?;
    Ok(())
}
