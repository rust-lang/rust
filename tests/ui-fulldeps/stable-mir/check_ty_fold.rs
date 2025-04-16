//@ run-pass
//! Test that users are able to use stable mir APIs to retrieve monomorphized types, and that
//! we have an error handling for trying to instantiate types with incorrect arguments.

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

use stable_mir::mir::{
    Body, FieldIdx, MirVisitor, Place, ProjectionElem,
    visit::{Location, PlaceContext},
};
use stable_mir::ty::{RigidTy, Ty, TyKind};
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_stable_mir() -> ControlFlow<()> {
    let main_fn = stable_mir::entry_fn();
    let body = main_fn.unwrap().expect_body();
    let mut visitor = PlaceVisitor { body: &body, tested: false };
    visitor.visit_body(&body);
    assert!(visitor.tested);
    ControlFlow::Continue(())
}

struct PlaceVisitor<'a> {
    body: &'a Body,
    /// Used to ensure that the test was reachable. Otherwise this test would vacuously succeed.
    tested: bool,
}

/// Check that `wrapper.inner` place projection can be correctly interpreted.
/// Ensure that instantiation is correct.
fn check_tys(local_ty: Ty, idx: FieldIdx, expected_ty: Ty) {
    let TyKind::RigidTy(RigidTy::Adt(def, args)) = local_ty.kind() else { unreachable!() };
    assert_eq!(def.ty_with_args(&args), local_ty);

    let field_def = &def.variants_iter().next().unwrap().fields()[idx];
    let field_ty = field_def.ty_with_args(&args);
    assert_eq!(field_ty, expected_ty);

    // Check that the generic version is different than the instantiated one.
    let field_ty_gen = field_def.ty();
    assert_ne!(field_ty_gen, field_ty);
}

impl<'a> MirVisitor for PlaceVisitor<'a> {
    fn visit_place(&mut self, place: &Place, _ptx: PlaceContext, _loc: Location) {
        let start_ty = self.body.locals()[place.local].ty;
        match place.projection.as_slice() {
            [ProjectionElem::Field(idx, ty)] => {
                check_tys(start_ty, *idx, *ty);
                self.tested = true;
            }
            _ => {}
        }
    }
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `StableMir` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "ty_fold_input.rs";
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
        struct Wrapper<T: Default> {{
            pub inner: T
        }}

        impl<T: Default> Wrapper<T> {{
            pub fn new() -> Wrapper<T> {{
                Wrapper {{ inner: T::default() }}
            }}
        }}

        fn main() {{
            let wrapper = Wrapper::<u8>::new();
            let _inner = wrapper.inner;
        }}
    "#
    )?;
    Ok(())
}
