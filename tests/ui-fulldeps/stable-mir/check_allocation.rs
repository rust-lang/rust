// run-pass
//! Test that users are able to use stable mir APIs to retrieve information of global allocations
//! such as `vtable_allocation`.

// ignore-stage1
// ignore-cross-compile
// ignore-remote
// ignore-windows-gnu mingw has troubles with linking https://github.com/rust-lang/rust/pull/116837
// edition: 2021

#![feature(rustc_private)]
#![feature(assert_matches)]
#![feature(control_flow_enum)]
#![feature(ascii_char, ascii_char_variants)]

extern crate rustc_hir;
extern crate rustc_middle;
#[macro_use]
extern crate rustc_smir;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate stable_mir;

use rustc_middle::ty::TyCtxt;
use rustc_smir::rustc_internal;
use stable_mir::{CrateItem, CrateItems, ItemKind};
use stable_mir::mir::alloc::GlobalAlloc;
use stable_mir::mir::mono::StaticDef;
use std::ascii::Char;
use std::assert_matches::assert_matches;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_stable_mir(_tcx: TyCtxt<'_>) -> ControlFlow<()> {
    // Find items in the local crate.
    let items = stable_mir::all_local_items();
    check_foo(*get_item(&items, (ItemKind::Static, "FOO")).unwrap());
    check_bar(*get_item(&items, (ItemKind::Static, "BAR")).unwrap());
    ControlFlow::Continue(())
}

/// Check the allocation data for static `FOO`.
///
/// ```no_run
/// static FOO: [&str; 2] = ["hi", "there"];
/// ```
fn check_foo(item: CrateItem) {
    let def = StaticDef::try_from(item).unwrap();
    let alloc = def.eval_initializer().unwrap();
    assert_eq!(alloc.provenance.ptrs.len(), 2);

    let alloc_id_0 = alloc.provenance.ptrs[0].1.0;
    assert_matches!(GlobalAlloc::from(alloc_id_0), GlobalAlloc::Memory(..));

    let alloc_id_1 = alloc.provenance.ptrs[1].1.0;
    assert_matches!(GlobalAlloc::from(alloc_id_1), GlobalAlloc::Memory(..));
}

/// Check the allocation data for static `BAR`.
///
/// ```no_run
/// static BAR: &str = "Bar";
/// ```
fn check_bar(item: CrateItem) {
    let def = StaticDef::try_from(item).unwrap();
    let alloc = def.eval_initializer().unwrap();
    assert_eq!(alloc.provenance.ptrs.len(), 1);

    let alloc_id_0 = alloc.provenance.ptrs[0].1.0;
    let GlobalAlloc::Memory(allocation) = GlobalAlloc::from(alloc_id_0) else { unreachable!() };
    assert_eq!(allocation.bytes.len(), 3);
    assert_eq!(allocation.bytes[0].unwrap(), Char::CapitalB.to_u8());
    assert_eq!(allocation.bytes[1].unwrap(), Char::SmallA.to_u8());
    assert_eq!(allocation.bytes[2].unwrap(), Char::SmallR.to_u8());
}

// Use internal API to find a function in a crate.
fn get_item<'a>(
    items: &'a CrateItems,
    item: (ItemKind, &str),
) -> Option<&'a stable_mir::CrateItem> {
    items.iter().find(|crate_item| {
        (item.0 == crate_item.kind()) && crate_item.name() == item.1
    })
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `StableMir` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "alloc_input.rs";
    generate_input(&path).unwrap();
    let args = vec![
        "rustc".to_string(),
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
    static FOO: [&str; 2] = ["hi", "there"];
    static BAR: &str = "Bar";

    pub fn main() {{
        println!("{{FOO:?}}! {{BAR}}");
    }}"#
    )?;
    Ok(())
}
