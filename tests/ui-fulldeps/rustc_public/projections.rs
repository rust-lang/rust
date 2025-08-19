//@ run-pass
// Tests the Stable MIR projections API

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2021

#![feature(rustc_private)]
#![feature(assert_matches)]

extern crate rustc_hir;
extern crate rustc_middle;

extern crate rustc_driver;
extern crate rustc_interface;
#[macro_use]
extern crate rustc_public;

use rustc_public::ItemKind;
use rustc_public::crate_def::CrateDef;
use rustc_public::mir::{ProjectionElem, Rvalue, StatementKind};
use rustc_public::ty::{RigidTy, TyKind, UintTy};
use std::assert_matches::assert_matches;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

/// Tests projections within Place objects
fn test_place_projections() -> ControlFlow<()> {
    let items = rustc_public::all_local_items();
    let body = get_item(&items, (ItemKind::Fn, "projections")).unwrap().expect_body();
    assert_eq!(body.blocks.len(), 4);
    // The first statement assigns `&s.c` to a local. The projections include a deref for `s`, since
    // `s` is passed as a reference argument, and a field access for field `c`.
    match &body.blocks[0].statements[0].kind {
        StatementKind::Assign(
            place @ rustc_public::mir::Place { local: _, projection: local_proj },
            Rvalue::Ref(_, _, rustc_public::mir::Place { local: _, projection: r_proj }),
        ) => {
            // We can't match on vecs, only on slices. Comparing statements for equality wouldn't be
            // any easier since we'd then have to add in the expected local and region values
            // instead of matching on wildcards.
            assert!(local_proj.is_empty());
            match &r_proj[..] {
                // Similarly we can't match against a type, only against its kind.
                [ProjectionElem::Deref, ProjectionElem::Field(2, ty)] => {
                    assert_matches!(
                        ty.kind(),
                        TyKind::RigidTy(RigidTy::Uint(rustc_public::ty::UintTy::U8))
                    );
                    let ty = place.ty(body.locals()).unwrap();
                    assert_matches!(ty.kind().rigid(), Some(RigidTy::Ref(..)));
                }
                other => panic!(
                    "Unable to match against expected rvalue projection. Expected the projection \
                     for `s.c`, which is a Deref and u8 Field. Got: {:?}",
                    other
                ),
            };
        }
        other => panic!(
            "Unable to match against expected Assign statement with a Ref rvalue. Expected the \
             statement to assign `&s.c` to a local. Got: {:?}",
            other
        ),
    };
    // This statement assigns `slice[1]` to a local. The projections include a deref for `slice`,
    // since `slice` is a reference, and an index.
    match &body.blocks[2].statements[0].kind {
        StatementKind::Assign(
            place @ rustc_public::mir::Place { local: _, projection: local_proj },
            Rvalue::Use(rustc_public::mir::Operand::Copy(rustc_public::mir::Place {
                local: _,
                projection: r_proj,
            })),
        ) => {
            // We can't match on vecs, only on slices. Comparing for equality wouldn't be any easier
            // since we'd then have to add in the expected local values instead of matching on
            // wildcards.
            assert!(local_proj.is_empty());
            assert_matches!(r_proj[..], [ProjectionElem::Deref, ProjectionElem::Index(_)]);
            let ty = place.ty(body.locals()).unwrap();
            assert_matches!(ty.kind().rigid(), Some(RigidTy::Uint(UintTy::U8)));
        }
        other => panic!(
            "Unable to match against expected Assign statement with a Use rvalue. Expected the \
             statement to assign `slice[1]` to a local. Got: {:?}",
            other
        ),
    };
    // The first terminator gets a slice of an array via the Index operation. Specifically it
    // performs `&vals[1..3]`. There are no projections in this case, the arguments are just locals.
    match &body.blocks[0].terminator.kind {
        rustc_public::mir::TerminatorKind::Call { args, .. } =>
        // We can't match on vecs, only on slices. Comparing for equality wouldn't be any easier
        // since we'd then have to add in the expected local values instead of matching on
        // wildcards.
        {
            match &args[..] {
                [
                    rustc_public::mir::Operand::Move(rustc_public::mir::Place {
                        local: _,
                        projection: arg1_proj,
                    }),
                    rustc_public::mir::Operand::Move(rustc_public::mir::Place {
                        local: _,
                        projection: arg2_proj,
                    }),
                ] => {
                    assert!(arg1_proj.is_empty());
                    assert!(arg2_proj.is_empty());
                }
                other => {
                    panic!(
                        "Unable to match against expected arguments to Index call. Expected two \
                         move operands. Got: {:?}",
                        other
                    )
                }
            }
        }
        other => panic!(
            "Unable to match against expected Call terminator. Expected a terminator that calls \
             the Index operation. Got: {:?}",
            other
        ),
    };

    ControlFlow::Continue(())
}

// Use internal API to find a function in a crate.
fn get_item<'a>(
    items: &'a rustc_public::CrateItems,
    item: (ItemKind, &str),
) -> Option<&'a rustc_public::CrateItem> {
    items.iter().find(|crate_item| crate_item.kind() == item.0 && crate_item.name() == item.1)
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `RustcPublic` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "input.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "--crate-type=lib".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args, test_place_projections).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
    pub struct Struct1 {{ _a: u8, _b: u16, c: u8 }}

    pub fn projections(s: &Struct1) -> u8 {{
        let v = &s.c;
        let vals = [1, 2, 3, 4];
        let slice = &vals[1..3];
        v + slice[1]
    }}"#
    )?;
    Ok(())
}
