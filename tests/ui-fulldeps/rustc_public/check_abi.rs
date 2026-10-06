//@ run-pass
//! Test information regarding type layout.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
#[macro_use]
extern crate rustc_public;
extern crate rustc_target;

use std::assert_matches;
use std::collections::HashSet;
use std::convert::TryFrom;
use std::io::Write;
use std::ops::ControlFlow;

use rustc_public::abi::{
    ArgAbi, ArgExtension, CallConvention, FieldsShape, IndirectMode, IntegerLength, PassMode,
    Primitive, Scalar, ValueRepr, VariantsShape,
};
use rustc_public::mir::MirVisitor;
use rustc_public::mir::mono::Instance;
use rustc_public::target::MachineInfo;
use rustc_public::ty::{AdtDef, RigidTy, Ty, TyKind};
use rustc_public::{CrateDef, CrateItem, CrateItems, ItemKind};

const CRATE_NAME: &str = "input";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_stable_mir() -> ControlFlow<()> {
    check_regular_attributes();

    // Find items in the local crate.
    let items = rustc_public::all_local_items();

    // Test fn_abi
    let target_fn =
        *get_item(&items, (ItemKind::Fn, "input::fn_abi")).expect("Expected fn_abi function");
    let instance = Instance::try_from(target_fn).expect("Expected fn_abi instance");
    let fn_abi = instance.fn_abi().expect("Expected function ABI");
    assert_eq!(fn_abi.conv, CallConvention::Rust, "Expected Rust calling convention for fn_abi");
    assert_eq!(fn_abi.args.len(), 3, "Expected three arguments for fn_abi");

    check_ignore(&fn_abi.args[0]);
    check_primitive(&fn_abi.args[1]);
    check_niche(&fn_abi.args[2]);
    check_result(&fn_abi.ret);

    // Test variadic function.
    let variadic_fn = *get_item(&items, (ItemKind::Fn, "input::variadic_fn"))
        .expect("Expected variadic_fn function");
    check_variadic(variadic_fn);

    // Extract function pointers.
    let fn_ptr_holder = *get_item(&items, (ItemKind::Fn, "input::fn_ptr_holder"))
        .expect("Expected fn_ptr_holder function");
    let fn_ptr_holder_instance =
        Instance::try_from(fn_ptr_holder).expect("Expected fn_ptr_holder instance");
    let body = fn_ptr_holder_instance.body().expect("Expected fn_ptr_holder body");
    let args = body.arg_locals();

    // Test fn_abi of function pointer version.
    let ptr_fn_abi = args[0]
        .ty
        .kind()
        .fn_sig()
        .expect("Expected ComplexFn signature")
        .fn_ptr_abi()
        .expect("Expected ComplexFn ABI");
    assert_eq!(ptr_fn_abi, fn_abi, "Expected matching function and function pointer ABIs");

    // Test variadic_fn of function pointer version.
    let ptr_variadic_fn_abi = args[1]
        .ty
        .kind()
        .fn_sig()
        .expect("Expected VariadicFn signature")
        .fn_ptr_abi()
        .expect("Expected VariadicFn ABI");
    assert!(ptr_variadic_fn_abi.c_variadic, "Expected C variadic function pointer");
    assert_eq!(ptr_variadic_fn_abi.args.len(), 1, "Expected one fixed variadic pointer argument");

    let entry = rustc_public::entry_fn().expect("Expected main function");
    let main_fn = Instance::try_from(entry).expect("Expected main instance");
    let mut visitor = AdtDefVisitor::default();
    visitor.visit_body(&main_fn.body().expect("Expected main body"));
    let AdtDefVisitor { adt_defs } = visitor;
    assert_eq!(adt_defs.len(), 1, "Expected one ADT in main");

    // Test ADT representation options
    let repr_c_struct = adt_defs
        .iter()
        .find(|def| def.trimmed_name() == "ReprCStruct")
        .expect("Expected ReprCStruct definition");
    assert!(repr_c_struct.repr().flags.is_c, "Expected repr(C) for ReprCStruct");

    ControlFlow::Continue(())
}

fn check_regular_attributes() {
    use rustc_target::callconv::{ArgAttribute, ArgAttributes};

    let flags = [
        ArgAttribute::CapturesNone,
        ArgAttribute::CapturesAddress,
        ArgAttribute::CapturesReadOnly,
        ArgAttribute::NoAlias,
        ArgAttribute::NonNull,
        ArgAttribute::ReadOnly,
        ArgAttribute::InReg,
        ArgAttribute::NoUndef,
        ArgAttribute::Writable,
        ArgAttribute::NoFree,
    ];
    for regular in [ArgAttribute::empty(), ArgAttribute::all()].iter().copied().chain(flags) {
        let attrs = rustc_public::rustc_internal::stable(ArgAttributes::from(regular)).regular();
        for (flag, actual) in flags.iter().copied().zip([
            attrs.captures_none,
            attrs.captures_address,
            attrs.captures_read_only,
            attrs.no_alias,
            attrs.non_null,
            attrs.read_only,
            attrs.in_reg,
            attrs.no_undef,
            attrs.writable,
            attrs.no_free,
        ]) {
            assert_eq!(actual, regular.contains(flag), "Expected {flag:?} for {regular:?}");
        }
    }
}

struct ExpectedArgAttributes {
    no_alias: bool,
    non_null: bool,
    read_only: bool,
    no_free: bool,
    pointee_size: usize,
}

fn check_args(args: &[ArgAbi], expected: &[(&str, ExpectedArgAttributes)]) {
    assert_eq!(args.len(), expected.len(), "Expected one set of attributes per argument");
    for (arg, (name, expected)) in args.iter().zip(expected) {
        let PassMode::Direct(attrs) = &arg.mode else {
            panic!("Expected PassMode::Direct for {name}, got: {:?}", arg.mode);
        };
        let regular = attrs.regular();
        assert_eq!(regular.no_alias, expected.no_alias, "Expected no_alias for {name}");
        assert_eq!(regular.non_null, expected.non_null, "Expected non_null for {name}");
        assert_eq!(regular.read_only, expected.read_only, "Expected read_only for {name}");
        assert_eq!(regular.no_free, expected.no_free, "Expected no_free for {name}");
        assert_eq!(
            attrs.pointee_size().bytes(),
            expected.pointee_size,
            "Expected pointee_size for {name}",
        );
        assert!(regular.no_undef, "Expected no_undef for {}", name);
        assert_eq!(
            regular.captures_read_only,
            expected.read_only,
            "Expected captures_read_only for {name}",
        );
        assert!(!regular.captures_none, "Expected captures_none to be unset for {}", name);
        assert!(!regular.captures_address, "Expected captures_address to be unset for {}", name);
    }
}

fn test_pointer_attributes() -> ControlFlow<()> {
    let items = rustc_public::all_local_items();
    let target_fn = get_item(&items, (ItemKind::Fn, "input::pointer_attributes"))
        .expect("Expected pointer_attributes function");
    // Use the signature to exclude attributes deduced from the body.
    let abi = target_fn
        .ty()
        .kind()
        .fn_sig()
        .expect("Expected pointer_attributes signature")
        .fn_ptr_abi()
        .expect("Expected pointer_attributes ABI");
    assert_eq!(abi.args.len(), 6, "Expected six arguments for pointer_attributes");

    check_args(
        &abi.args[..4],
        &[
            (
                "shared",
                ExpectedArgAttributes {
                    no_alias: true,
                    non_null: true,
                    read_only: true,
                    no_free: true,
                    pointee_size: 4,
                },
            ),
            (
                "cell",
                ExpectedArgAttributes {
                    no_alias: false,
                    non_null: true,
                    read_only: false,
                    no_free: false,
                    pointee_size: 4,
                },
            ),
            (
                "raw",
                ExpectedArgAttributes {
                    no_alias: false,
                    non_null: false,
                    read_only: false,
                    no_free: false,
                    pointee_size: 0,
                },
            ),
            (
                "nullable",
                ExpectedArgAttributes {
                    no_alias: true,
                    non_null: false,
                    read_only: true,
                    no_free: true,
                    pointee_size: 4,
                },
            ),
        ],
    );

    let PassMode::Pair(data, len) = &abi.args[4].mode else {
        panic!("Expected PassMode::Pair for slice, got: {:?}", abi.args[4].mode);
    };
    assert!(data.regular().non_null, "Expected non_null for slice data");
    assert!(data.regular().no_free, "Expected no_free for slice data");
    assert!(data.regular().no_alias, "Expected no_alias for slice data");
    assert!(data.regular().read_only, "Expected read_only for slice data");
    assert!(data.regular().captures_read_only, "Expected captures_read_only for slice data");
    assert!(len.regular().no_undef, "Expected no_undef for slice length");
    assert!(!len.regular().non_null, "Expected non_null to be unset for slice length");

    let PassMode::Indirect { attrs, .. } = &abi.args[5].mode else {
        panic!("Expected PassMode::Indirect for indirect, got: {:?}", abi.args[5].mode);
    };
    let regular = attrs.regular();
    assert!(regular.no_alias, "Expected no_alias for indirect");
    assert!(regular.non_null, "Expected non_null for indirect");
    assert!(regular.no_undef, "Expected no_undef for indirect");
    assert!(regular.no_free, "Expected no_free for indirect");
    assert!(!regular.captures_none, "Expected captures_none to be unset for indirect");
    assert!(regular.captures_address, "Expected captures_address for indirect");
    assert!(regular.captures_read_only, "Expected captures_read_only for indirect");
    assert_eq!(attrs.pointee_size().bytes(), 256, "Expected pointee_size for indirect");

    let PassMode::Direct(attrs) = &abi.ret.mode else {
        panic!("Expected PassMode::Direct for return value, got: {:?}", abi.ret.mode);
    };
    let regular = attrs.regular();
    assert!(regular.non_null, "Expected non_null for return value");
    assert!(regular.no_undef, "Expected no_undef for return value");
    assert!(!regular.no_alias, "Expected no_alias to be unset for return value");
    assert!(!regular.read_only, "Expected read_only to be unset for return value");
    assert!(!regular.no_free, "Expected no_free to be unset for return value");
    assert!(!regular.captures_none, "Expected captures_none to be unset for return value");
    assert!(!regular.captures_address, "Expected captures_address to be unset for return value");
    assert!(
        !regular.captures_read_only,
        "Expected captures_read_only to be unset for return value"
    );
    assert_eq!(attrs.pointee_size().bytes(), 4, "Expected pointee_size for return value");
    ControlFlow::Continue(())
}

/// Check the variadic function ABI:
/// ```no_run
/// pub unsafe extern "C" fn variadic_fn(n: usize, mut args: ...) -> usize {
///     0
/// }
/// ```
fn check_variadic(variadic_fn: CrateItem) {
    let instance = Instance::try_from(variadic_fn).expect("Expected variadic_fn instance");
    let abi = instance.fn_abi().expect("Expected function ABI");
    assert!(abi.c_variadic, "Expected C variadic function");
    assert_eq!(abi.args.len(), 1, "Expected one fixed variadic argument");
}

/// Check the argument to be ignored: `ignore: [u8; 0]`.
fn check_ignore(abi: &ArgAbi) {
    assert!(abi.ty.kind().is_array(), "Expected array argument");
    assert_eq!(abi.mode, PassMode::Ignore, "Expected PassMode::Ignore for empty array");
    let layout = abi.layout.shape();
    assert!(layout.is_sized(), "Expected sized layout for empty array");
    assert!(layout.is_1zst(), "Expected empty array to be a 1-ZST");
}

/// Check the primitive argument: `primitive: char`.
fn check_primitive(abi: &ArgAbi) {
    assert!(abi.ty.kind().is_char(), "Expected char argument");
    let PassMode::Direct(ref attrs) = abi.mode else {
        panic!("Expected PassMode::Direct for char, got: {:?}", abi.mode);
    };
    // A char (32-bit) doesn't need sign/zero extension on most platforms.
    #[cfg(not(any(target_arch = "loongarch64", target_arch = "riscv64")))]
    assert_eq!(attrs.arg_extension(), ArgExtension::None, "Expected no extension for char");
    // However, LoongArch64 and RiscV64 ABIs require that 32-bit integers
    // (signed or unsigned) are sign-extended when passed in registers.
    #[cfg(any(target_arch = "loongarch64", target_arch = "riscv64"))]
    assert_eq!(attrs.arg_extension(), ArgExtension::Sext, "Expected sign extension for char");
    // Direct arguments are not pointers, so no pointee alignment.
    assert_eq!(attrs.pointee_align(), None, "Expected no pointee alignment for char");
    let layout = abi.layout.shape();
    assert!(layout.is_sized(), "Expected sized layout for char");
    assert!(!layout.is_1zst(), "Expected char not to be a 1-ZST");
    assert_matches!(layout.fields, FieldsShape::Primitive, "Expected primitive fields for char");
}

/// Check the return value: `Result<usize, &str>`.
fn check_result(abi: &ArgAbi) {
    assert!(abi.ty.kind().is_enum(), "Expected Result enum");
    let PassMode::Indirect { ref attrs, address_space: _, mode } = abi.mode else {
        panic!("Expected PassMode::Indirect for Result, got: {:?}", abi.mode);
    };
    // Indirect arguments have a pointee alignment (the pointer must be aligned).
    assert!(attrs.pointee_align().is_some(), "Expected pointee alignment for Result");
    assert_eq!(mode, IndirectMode::Pointer, "Expected indirect pointer for Result");
    let layout = abi.layout.shape();
    assert!(layout.is_sized(), "Expected sized layout for Result");
    assert_matches!(
        layout.fields,
        FieldsShape::Arbitrary { .. },
        "Expected arbitrary fields for Result",
    );
    assert_matches!(
        layout.variants,
        VariantsShape::Multiple { .. },
        "Expected multiple variants for Result",
    );
}

/// Checks the niche information about `NonZero<u8>`.
fn check_niche(abi: &ArgAbi) {
    assert!(abi.ty.kind().is_struct(), "Expected NonZero<u8> struct");
    assert_matches!(abi.mode, PassMode::Direct { .. }, "Expected PassMode::Direct for NonZero<u8>");
    let layout = abi.layout.shape();
    assert!(layout.is_sized(), "Expected sized layout for NonZero<u8>");
    assert_eq!(layout.size.bytes(), 1, "Expected one-byte size for NonZero<u8>");

    let ValueRepr::Scalar(scalar) = layout.value_repr else {
        panic!("Expected scalar representation for NonZero<u8>, got: {:?}", layout.value_repr);
    };
    assert!(
        scalar.has_niche(&MachineInfo::target()),
        "Expected niche for NonZero<u8>, got: {:?}",
        scalar,
    );

    let Scalar::Initialized { value, valid_range } = scalar else {
        panic!("Expected initialized scalar for NonZero<u8>, got: {:?}", scalar);
    };
    assert_matches!(
        value,
        Primitive::Int { length: IntegerLength::I8, signed: false },
        "Expected u8 scalar for NonZero<u8>",
    );
    assert_eq!(valid_range.start, 1, "Expected valid range to start at one for NonZero<u8>");
    assert_eq!(
        valid_range.end,
        u8::MAX.into(),
        "Expected valid range to end at u8::MAX for NonZero<u8>"
    );
    assert!(!valid_range.contains(0), "Expected zero outside valid range for NonZero<u8>");
    assert!(!valid_range.wraps_around(), "Expected non-wrapping valid range for NonZero<u8>");
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
    generate_input(&path).expect("Expected input file to be generated");
    let mut args = vec![
        "rustc".to_string(),
        "-Cpanic=abort".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(&args, test_stable_mir).expect("Expected ABI checks to succeed");
    args.push("-Copt-level=1".to_string());
    run!(&args, test_pointer_attributes).expect("Expected pointer attribute checks to succeed");
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
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

        pub fn pointer_attributes<'a>(
            shared: &'a u32,
            cell: &std::cell::Cell<u32>,
            raw: *const u32,
            nullable: Option<&u32>,
            slice: &[u32],
            indirect: [u64; 32],
        ) -> &'a u32 {{
            shared
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
