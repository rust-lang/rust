//@ run-pass
//! Test that users are able to retrieve information about ADTs

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2024

#![feature(rustc_private)]

extern crate rustc_middle;

extern crate rustc_driver;
extern crate rustc_interface;
#[macro_use]
extern crate rustc_public;

use std::{collections::HashSet, io::Write, ops::ControlFlow};
use rustc_public::{
    mir::{mono::Instance, MirVisitor},
    ty::{AdtDef, RigidTy, Ty, TyKind}
};

const CRATE_NAME: &str = "adt_test";

#[derive(Default)]
struct AdtDefVisitor {
    adt_defs: HashSet<AdtDef>,
}

impl MirVisitor for AdtDefVisitor {
    fn visit_ty(&mut self, ty: &Ty, _location: rustc_public::mir::visit::Location) {
        if let TyKind::RigidTy(RigidTy::Adt(adt, _)) = ty.kind() {
            self.adt_defs.insert(adt);
        }
        self.super_ty(ty)
    }
}

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_adts() -> ControlFlow<()> {
    let entry = rustc_public::entry_fn().expect("Expected main function");
    let main_fn = Instance::try_from(entry).expect("Expected main instance");
    let mut visitor = AdtDefVisitor::default();
    visitor.visit_body(&main_fn.body().expect("Expected main body"));
    let AdtDefVisitor { adt_defs } = visitor;

    assert_eq!(
        adt_defs.len(),
        5,
        "Expected 5 ADTs, but found {:?}", adt_defs,
    );

    let has_box = adt_defs
        .iter()
        .any(|item| item.is_box());

    let has_simd = adt_defs
        .iter()
        .any(|item| item.is_simd());

    let has_phantom_data = adt_defs
        .iter()
        .any(|item| item.is_phantom_data());

    let has_manually_drop = adt_defs
        .iter()
        .any(|item| item.is_manually_drop());

    let has_unsafe_cell = adt_defs
        .iter()
        .any(|item| item.is_unsafe_cell());

    assert!(has_box);
    assert!(has_simd);
    assert!(has_phantom_data);
    assert!(has_manually_drop);
    assert!(has_unsafe_cell);

    let adt_names = adt_defs.iter().map(|def| def.0.trimmed_name()).collect::<HashSet<_>>();

    assert_name(&adt_names, "Box");
    assert_name(&adt_names, "PhantomData");
    assert_name(&adt_names, "Simd");
    assert_name(&adt_names, "ManuallyDrop");
    assert_name(&adt_names, "UnsafeCell");

    // Verify all AdtDef::ty() returns the self type (Adt)
    adt_defs.iter().for_each(|adt|{
        let ty = adt.ty();
        assert!(
            matches!(
                ty.kind(),
                rustc_public::ty::TyKind::RigidTy(RigidTy::Adt(..))
            ),
        );
    });

    ControlFlow::Continue(())
}

fn assert_name(adt_names: &HashSet<String>, target: &str) {
    assert!(
        adt_names.contains(target),
        "Failed to find `{target}`. Implementations available: {adt_names:?}",
    );
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `RustcPublic` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "adt_queries.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "-Cpanic=abort".to_string(),
        "--edition".to_string(),
        "2024".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args, test_adts).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
        #![feature(portable_simd)]

        use std::boxed::Box;
        use std::cell::UnsafeCell;
        use std::ffi::CStr;
        use std::mem::ManuallyDrop;
        use std::marker::PhantomData;
        use std::simd::f32x4;

        fn main() {{
            let _box = Box::new(42);

            let _phantom = PhantomData::<u64>;

            let _unsafe_cell = UnsafeCell::new(42);

            let mut _manually_drop = ManuallyDrop::new(42);

            let _cs: &CStr = c"hello";

            let _simd = f32x4::splat(10.0);
        }}
    "#
    )?;
    Ok(())
}
