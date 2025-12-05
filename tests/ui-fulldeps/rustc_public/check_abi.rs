//@ run-pass
//! Test information regarding type layout.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote

#![feature(rustc_private)]
#![feature(assert_matches)]
#![feature(ascii_char, ascii_char_variants)]

extern crate rustc_hir;
extern crate rustc_middle;
extern crate rustc_driver;
extern crate rustc_interface;
#[macro_use]
extern crate rustc_public;

use rustc_public::abi::{
    ArgAbi, CallConvention, FieldsShape, IntegerLength, PassMode, Primitive, Scalar, ValueAbi,
    VariantsShape,
};
use rustc_public::mir::MirVisitor;
use rustc_public::mir::mono::Instance;
use rustc_public::target::MachineInfo;
use rustc_public::ty::{AdtDef, RigidTy, Ty, TyKind};
use rustc_public::{CrateDef, CrateItem, CrateItems, ItemKind};
use std::assert_matches::assert_matches;
use std::collections::HashSet;
use std::convert::TryFrom;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_stable_mir() -> ControlFlow<()> {
    // Find items in the local crate.
    let items = rustc_public::all_local_items();

    // Test fn_abi
    let target_fn = *get_item(&items, (ItemKind::Fn, "fn_abi")).unwrap();
    let instance = Instance::try_from(target_fn).unwrap();
    let fn_abi = instance.fn_abi().unwrap();
    assert_eq!(fn_abi.conv, CallConvention::Rust);
    assert_eq!(fn_abi.args.len(), 3);

    check_ignore(&fn_abi.args[0]);
    check_primitive(&fn_abi.args[1]);
    check_niche(&fn_abi.args[2]);
    check_result(&fn_abi.ret);

    // Test variadic function.
    let variadic_fn = *get_item(&items, (ItemKind::Fn, "variadic_fn")).unwrap();
    check_variadic(variadic_fn);

    // Extract function pointers.
    let fn_ptr_holder = *get_item(&items, (ItemKind::Fn, "fn_ptr_holder")).unwrap();
    let fn_ptr_holder_instance = Instance::try_from(fn_ptr_holder).unwrap();
    let body = fn_ptr_holder_instance.body().unwrap();
    let args = body.arg_locals();

    // Test fn_abi of function pointer version.
    let ptr_fn_abi = args[0].ty.kind().fn_sig().unwrap().fn_ptr_abi().unwrap();
    assert_eq!(ptr_fn_abi, fn_abi);

    // Test variadic_fn of function pointer version.
    let ptr_variadic_fn_abi = args[1].ty.kind().fn_sig().unwrap().fn_ptr_abi().unwrap();
    assert!(ptr_variadic_fn_abi.c_variadic);
    assert_eq!(ptr_variadic_fn_abi.args.len(), 1);

    let entry = rustc_public::entry_fn().unwrap();
    let main_fn = Instance::try_from(entry).unwrap();
    let mut visitor = AdtDefVisitor::default();
    visitor.visit_body(&main_fn.body().unwrap());
    let AdtDefVisitor { adt_defs } = visitor;
    assert_eq!(adt_defs.len(), 1);

    // Test ADT representation options
    let repr_c_struct = adt_defs.iter().find(|def| def.trimmed_name() == "ReprCStruct").unwrap();
    assert!(repr_c_struct.repr().flags.is_c);

    ControlFlow::Continue(())
}

/// Check the variadic function ABI:
/// ```no_run
/// pub unsafe extern "C" fn variadic_fn(n: usize, mut args: ...) -> usize {
///     0
/// }
/// ```
fn check_variadic(variadic_fn: CrateItem) {
    let instance = Instance::try_from(variadic_fn).unwrap();
    let abi = instance.fn_abi().unwrap();
    assert!(abi.c_variadic);
    assert_eq!(abi.args.len(), 1);
}

/// Check the argument to be ignored: `ignore: [u8; 0]`.
fn check_ignore(abi: &ArgAbi) {
    assert!(abi.ty.kind().is_array());
    assert_eq!(abi.mode, PassMode::Ignore);
    let layout = abi.layout.shape();
    assert!(layout.is_sized());
    assert!(layout.is_1zst());
}

/// Check the primitive argument: `primitive: char`.
fn check_primitive(abi: &ArgAbi) {
    assert!(abi.ty.kind().is_char());
    assert_matches!(abi.mode, PassMode::Direct(_));
    let layout = abi.layout.shape();
    assert!(layout.is_sized());
    assert!(!layout.is_1zst());
    assert_matches!(layout.fields, FieldsShape::Primitive);
}

/// Check the return value: `Result<usize, &str>`.
fn check_result(abi: &ArgAbi) {
    assert!(abi.ty.kind().is_enum());
    assert_matches!(abi.mode, PassMode::Indirect { .. });
    let layout = abi.layout.shape();
    assert!(layout.is_sized());
    assert_matches!(layout.fields, FieldsShape::Arbitrary { .. });
    assert_matches!(layout.variants, VariantsShape::Multiple { .. })
}

/// Checks the niche information about `NonZero<u8>`.
fn check_niche(abi: &ArgAbi) {
    assert!(abi.ty.kind().is_struct());
    assert_matches!(abi.mode, PassMode::Direct { .. });
    let layout = abi.layout.shape();
    assert!(layout.is_sized());
    assert_eq!(layout.size.bytes(), 1);

    let ValueAbi::Scalar(scalar) = layout.abi else { unreachable!() };
    assert!(scalar.has_niche(&MachineInfo::target()), "Opps: {:?}", scalar);

    let Scalar::Initialized { value, valid_range } = scalar else { unreachable!() };
    assert_matches!(value, Primitive::Int { length: IntegerLength::I8, signed: false });
    assert_eq!(valid_range.start, 1);
    assert_eq!(valid_range.end, u8::MAX.into());
    assert!(!valid_range.contains(0));
    assert!(!valid_range.wraps_around());
}

fn get_item<'a>(
    items: &'a CrateItems,
    item: (ItemKind, &str),
) -> Option<&'a rustc_public::CrateItem> {
    items.iter().find(|crate_item| (item.0 == crate_item.kind()) && crate_item.name() == item.1)
}

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

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `RustcPublic` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "alloc_input.rs";
    generate_input(&path).unwrap();
    let args = &[
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
        #![feature(c_variadic)]
        #![allow(unused_variables)]

        use std::num::NonZero;

        pub fn fn_abi(
            ignore: [u8; 0],
            primitive: char,
            niche: NonZero<u8>,
        ) -> Result<usize, &'static str> {{
                // We only care about the signature.
                todo!()
        }}

        pub unsafe extern "C" fn variadic_fn(n: usize, mut args: ...) -> usize {{
            0
        }}

        pub type ComplexFn = fn([u8; 0], char, NonZero<u8>) -> Result<usize, &'static str>;
        pub type VariadicFn = unsafe extern "C" fn(usize, ...) -> usize;

        pub fn fn_ptr_holder(complex_fn: ComplexFn, variadic_fn: VariadicFn) {{
            // We only care about the signature.
            todo!()
        }}

        fn main() {{
            #[repr(C)]
            struct ReprCStruct;

            let _s = ReprCStruct;
        }}
        "#
    )?;
    Ok(())
}
