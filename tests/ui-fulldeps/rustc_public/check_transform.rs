//@ run-pass
//! Test a few methods to transform StableMIR.

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

use rustc_public::mir::alloc::GlobalAlloc;
use rustc_public::mir::mono::Instance;
use rustc_public::mir::{Body, ConstOperand, Operand, Rvalue, StatementKind, TerminatorKind};
use rustc_public::ty::{ConstantKind, MirConst};
use rustc_public::{CrateDef, CrateItems, ItemKind};
use std::convert::TryFrom;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

/// This function uses the Stable MIR APIs to transform the MIR.
fn test_transform() -> ControlFlow<()> {
    // Find items in the local crate.
    let items = rustc_public::all_local_items();

    // Test fn_abi
    let target_fn = *get_item(&items, (ItemKind::Fn, "dummy")).unwrap();
    let instance = Instance::try_from(target_fn).unwrap();
    let body = instance.body().unwrap();
    check_msg(&body, "oops");

    let new_msg = "new panic message";
    let new_body = change_panic_msg(body, new_msg);
    check_msg(&new_body, new_msg);

    ControlFlow::Continue(())
}

/// Check that the body panic message matches the given message.
fn check_msg(body: &Body, expected: &str) {
    let msg = body
        .blocks
        .iter()
        .find_map(|bb| match &bb.terminator.kind {
            TerminatorKind::Call { args, .. } => {
                assert_eq!(args.len(), 1, "Expected panic message, but found {args:?}");
                let msg_const = match &args[0] {
                    Operand::Constant(msg_const) => msg_const,
                    Operand::Copy(place) | Operand::Move(place) => {
                        assert!(place.projection.is_empty());
                        bb.statements
                            .iter()
                            .find_map(|stmt| match &stmt.kind {
                                StatementKind::Assign(
                                    destination,
                                    Rvalue::Use(Operand::Constant(msg_const)),
                                ) if destination == place => Some(msg_const),
                                _ => None,
                            })
                            .unwrap()
                    }
                };
                let ConstantKind::Allocated(alloc) = msg_const.const_.kind() else {
                    unreachable!()
                };
                assert_eq!(alloc.provenance.ptrs.len(), 1);

                let alloc_prov_id = alloc.provenance.ptrs[0].1.0;
                let GlobalAlloc::Memory(val) = GlobalAlloc::from(alloc_prov_id) else {
                    unreachable!()
                };
                let bytes = val.raw_bytes().unwrap();
                Some(std::str::from_utf8(&bytes).unwrap().to_string())
            }
            _ => None,
        })
        .expect("Failed to find panic message");
    assert_eq!(&msg, expected);
}

/// Modify body to use a different panic message.
fn change_panic_msg(mut body: Body, new_msg: &str) -> Body {
    for bb in &mut body.blocks {
        match &mut bb.terminator.kind {
            TerminatorKind::Call { args, .. } => {
                let new_const = MirConst::from_str(new_msg);
                args[0] = Operand::Constant(ConstOperand {
                    const_: new_const,
                    span: bb.terminator.span,
                    user_ty: None,
                });
            }
            _ => {}
        }
    }
    body
}

fn get_item<'a>(
    items: &'a CrateItems,
    item: (ItemKind, &str),
) -> Option<&'a rustc_public::CrateItem> {
    items.iter().find(|crate_item| (item.0 == crate_item.kind()) && crate_item.name() == item.1)
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `RustcPublic` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "transform_input.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "--crate-type=lib".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args, test_transform).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
        fn panic_str(msg: &str) {{ panic!("{{}}", msg); }}
        pub fn dummy() {{
            panic_str("oops");
        }}
        "#
    )?;
    Ok(())
}
